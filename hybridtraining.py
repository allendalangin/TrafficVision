import mindspore as ms
from mindspore import nn, ops, train
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms import TypeCast
import numpy as np

# Import your custom modules
from src.coco_dataset import CocoDataset
from src.detector import TrafficVisionDetector

# --- Loss Function and Training Wrapper ---
class WithLossCell(nn.Cell):
    """Wraps the network with a loss function."""
    def __init__(self, network, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, image, gt_boxes, gt_labels):
        box_preds, class_preds = self.network(image)
        return self.loss_fn(box_preds, class_preds, gt_boxes, gt_labels)

# --- FINAL, STABLE DetectionLoss with CPU Matching and Sum Reduction ---
class DetectionLoss(nn.Cell):
    """
    A compatible loss function that performs matching on the CPU with NumPy
    and uses 'sum' reduction to ensure a stable gradient graph.
    """
    def __init__(self):
        super().__init__()
        # --- FIX: Change reduction to 'sum' for stability ---
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')

    def _calculate_iou_numpy(self, boxes1, boxes2):
        """Calculates IoU matrix using NumPy."""
        boxes1 = np.expand_dims(boxes1, 1)
        boxes2 = np.expand_dims(boxes2, 0)
        
        inter_top_left = np.maximum(boxes1[..., :2], boxes2[..., :2])
        inter_bottom_right = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        
        inter_wh = np.maximum(inter_bottom_right - inter_top_left, 0)
        intersection = inter_wh[..., 0] * inter_wh[..., 1]

        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-6)
        return iou

    def construct(self, box_preds, class_preds, gt_boxes, gt_labels):
        batch_size = box_preds.shape[0]
        box_preds_reshaped = box_preds.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 4))
        class_preds_reshaped = class_preds.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 9))

        total_reg_loss = ms.Tensor(0.0, dtype=ms.float32)
        total_cls_loss = ms.Tensor(0.0, dtype=ms.float32)
        total_matched_objects = 0

        for i in range(batch_size):
            # Move data to CPU for safe processing
            preds_boxes_np = box_preds_reshaped[i].asnumpy()
            preds_classes_np = class_preds_reshaped[i].asnumpy()
            gt_boxes_np = gt_boxes[i].asnumpy()
            gt_labels_np = gt_labels[i].asnumpy()
            
            # Perform matching in NumPy
            valid_gt_mask = gt_labels_np > -1
            valid_gt_boxes = gt_boxes_np[valid_gt_mask]
            valid_gt_labels = gt_labels_np[valid_gt_mask]
            
            num_gt_objects = valid_gt_boxes.shape[0]
            if num_gt_objects == 0:
                continue

            iou_matrix = self._calculate_iou_numpy(preds_boxes_np, valid_gt_boxes)
            best_match_idx = np.argmax(iou_matrix, axis=0)
            
            matched_pred_boxes_np = preds_boxes_np[best_match_idx]
            matched_pred_classes_np = preds_classes_np[best_match_idx]

            # Convert final pairs back to Tensors for loss calculation
            matched_pred_boxes = ms.Tensor(matched_pred_boxes_np)
            matched_pred_classes = ms.Tensor(matched_pred_classes_np)
            valid_gt_boxes_ms = ms.Tensor(valid_gt_boxes)
            valid_gt_labels_ms = ms.Tensor(valid_gt_labels)

            # Accumulate the SUM of losses
            total_reg_loss += self.smooth_l1_loss(matched_pred_boxes, valid_gt_boxes_ms)
            total_cls_loss += self.cross_entropy_loss(matched_pred_classes, valid_gt_labels_ms)
            total_matched_objects += num_gt_objects

        if total_matched_objects == 0:
            return ms.Tensor(0.0, dtype=ms.float32)

        # --- FIX: Simplify final loss calculation ---
        final_loss = (total_reg_loss + total_cls_loss) / total_matched_objects
        return final_loss

# --- The Reliable Manual Data Pipeline Generator ---
def create_batched_generator(source_dataset, batch_size, image_transforms, num_samples_to_use):
    """A compatible manual generator that handles the entire data pipeline."""
    print(f"[Generator] Preparing to use a random subset of {num_samples_to_use} images.")
    all_indices = np.arange(len(source_dataset))
    np.random.shuffle(all_indices)
    indices_to_use = all_indices[:num_samples_to_use]
    np.random.shuffle(indices_to_use)
    
    batch_buffer = []
    for i, idx in enumerate(indices_to_use):
        print(f"  [Generator] Preparing... Loaded {i+1}/{num_samples_to_use} images.", end='\r')
        image, boxes, labels = source_dataset[idx]
        
        transformed_image = image
        for op in image_transforms:
            transformed_image = op(transformed_image)
        batch_buffer.append((transformed_image, boxes, labels))
        
        if len(batch_buffer) == batch_size:
            images_list = [item[0] for item in batch_buffer]
            boxes_list = [item[1] for item in batch_buffer]
            labels_list = [item[2] for item in batch_buffer]
            
            max_objects = 0
            for b in boxes_list:
                if b.shape[0] > max_objects: max_objects = b.shape[0]
            if max_objects == 0: max_objects = 1

            padded_boxes, padded_labels = [], []
            for item_idx in range(batch_size):
                box_array = boxes_list[item_idx]
                padded_box = np.zeros((max_objects, 4), dtype=np.float32)
                if box_array.shape[0] > 0:
                    padded_box[:box_array.shape[0], :] = box_array
                padded_boxes.append(padded_box)

                label_array = labels_list[item_idx]
                padded_label = np.full((max_objects,), -1, dtype=np.int32)
                if label_array.shape[0] > 0:
                    padded_label[:label_array.shape[0]] = label_array
                padded_labels.append(padded_label)

            yield (np.stack(images_list), np.stack(padded_boxes), np.stack(padded_labels))
            batch_buffer = []

# --- Custom Callback for Monitoring ---
class StepAndEpochMonitor(train.Callback):
    """A custom callback to monitor training progress."""
    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        print(f"\n--- Epoch #{cb_params.cur_epoch_num} Begins ---")
    def step_begin(self, run_context):
        cb_params = run_context.original_args()
        print(f"  [Trainer] Step #{cb_params.cur_step_num} Begins...", end='\r')
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        print(f"\n--- Epoch #{cb_params.cur_epoch_num} Finished ---")

# --- Main Training Execution ---
if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")

    print("Initializing COCO dataset...")
    annotations_path = './coco2017/annotations/instances_train2017.json'
    images_path = './coco2017/train2017'
    coco_ds = CocoDataset(annotations_file=annotations_path, images_dir=images_path)

    # --- Pipeline Configuration ---
    BATCH_SIZE = 8
    NUM_SAMPLES = 36000
    NUM_EPOCHS = 20
    
    image_transforms = [
        vision.RandomHorizontalFlip(prob=0.5),
        vision.Resize((224, 224)),
        vision.HWC2CHW(),
        TypeCast(ms.float32)
    ]
    
    dataset = GeneratorDataset(
        source=lambda: create_batched_generator(coco_ds, BATCH_SIZE, image_transforms, NUM_SAMPLES),
        column_names=["image", "boxes", "labels"],
        num_parallel_workers=1
    )
    
    print("Reliable dataset pipeline created.")

    print("\nCreating model, loss function, and optimizer...")
    model = TrafficVisionDetector(num_classes=8)
    
    from mindspore import amp
    model = amp.auto_mixed_precision(model, amp_level="O2")

    loss_fn = DetectionLoss()
    network_with_loss = WithLossCell(model, loss_fn)

    steps_per_epoch = NUM_SAMPLES // BATCH_SIZE
    total_steps = steps_per_epoch * NUM_EPOCHS
    lr_schedule = nn.cosine_decay_lr(min_lr=0.0, max_lr=0.0001, total_step=total_steps,
                                     step_per_epoch=steps_per_epoch, decay_epoch=NUM_EPOCHS)
    
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr_schedule)

    trainer = train.Model(network=network_with_loss, optimizer=optimizer)

    loss_monitor = train.LossMonitor(per_print_times=10)
    checkpoint_cb = train.ModelCheckpoint(prefix="traffic_vision_36k", directory="./checkpoints")
    step_monitor = StepAndEpochMonitor()

    print(f"\n--- Starting Training on {NUM_SAMPLES} images for {NUM_EPOCHS} epochs ---")
    trainer.train(
        epoch=NUM_EPOCHS,
        train_dataset=dataset,
        callbacks=[loss_monitor, checkpoint_cb, step_monitor]
    )

    print("\n--- Training Finished ---")