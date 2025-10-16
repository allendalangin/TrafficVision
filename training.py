import mindspore as ms
from mindspore import nn, ops, train
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.vision as vision
import numpy as np
import time

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

class DetectionLoss(nn.Cell):
    """Fully vectorized loss function for object detection."""
    def __init__(self):
        super().__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')

    def construct(self, box_preds, class_preds, gt_boxes, gt_labels):
        batch_size = box_preds.shape[0]
        box_preds_reshaped = box_preds.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 4))
        class_preds_reshaped = class_preds.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 9))
        is_valid_label = (gt_labels > -1)
        num_gt_objects_per_image = ops.cast(is_valid_label, ms.int32).sum(axis=1)
        total_gt_objects = num_gt_objects_per_image.sum()
        
        if total_gt_objects == 0:
            return ms.Tensor(0.0, dtype=ms.float32)

        max_objects_in_batch = gt_labels.shape[1]
        range_tensor = ops.arange(max_objects_in_batch)
        mask = range_tensor < num_gt_objects_per_image.expand_dims(1)
        gt_boxes_masked = gt_boxes[mask]
        gt_labels_masked = gt_labels[mask]
        box_preds_to_consider = box_preds_reshaped[:, :max_objects_in_batch, :]
        class_preds_to_consider = class_preds_reshaped[:, :max_objects_in_batch, :]
        box_preds_masked = box_preds_to_consider[mask]
        class_preds_masked = class_preds_to_consider[mask]
        loss_reg = self.smooth_l1_loss(box_preds_masked, gt_boxes_masked)
        loss_cls = self.cross_entropy_loss(class_preds_masked, gt_labels_masked)
        total_loss = (loss_reg + loss_cls) / total_gt_objects
        return total_loss

# --- Fast In-Memory Generator ---
def create_preloaded_generator(processed_data, batch_size):
    """
    A fast generator that creates batches from data already loaded in RAM.
    """
    data_size = len(processed_data)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    for start_idx in range(0, data_size, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > data_size:
            continue # Drop remainder

        batch_indices = indices[start_idx:end_idx]
        
        # Get data for the current batch
        batch_images = [processed_data[i][0] for i in batch_indices]
        batch_boxes = [processed_data[i][1] for i in batch_indices]
        batch_labels = [processed_data[i][2] for i in batch_indices]
        
        # --- Perform Padding ---
        max_objects = 0
        for b in batch_boxes:
            if b.shape[0] > max_objects:
                max_objects = b.shape[0]
        if max_objects == 0:
            max_objects = 1

        padded_boxes = []
        padded_labels = []
        for i in range(batch_size):
            box_array = batch_boxes[i]
            padded_box = np.zeros((max_objects, 4), dtype=np.float32)
            if box_array.shape[0] > 0:
                padded_box[:box_array.shape[0], :] = box_array
            padded_boxes.append(padded_box)

            label_array = batch_labels[i]
            padded_label = np.full((max_objects,), -1, dtype=np.int32)
            if label_array.shape[0] > 0:
                padded_label[:label_array.shape[0]] = label_array
            padded_labels.append(padded_label)

        # Yield the final, padded batch
        yield (np.stack(batch_images), np.stack(padded_boxes), np.stack(padded_labels))

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

    # --- HYBRID PIPELINE CONFIGURATION ---
    BATCH_SIZE = 16 
    
    image_transforms = [
        vision.Resize((224, 224)),
        vision.HWC2CHW(),
    ]

    # 1. Pre-process the entire dataset into a list in RAM
    print("Pre-processing and loading all images into RAM. This may take a while...")
    start_time = time.time()
    processed_data = []
    for i, data in enumerate(coco_ds):
        image, boxes, labels = data
        # Apply transforms manually
        transformed_image = image
        for op in image_transforms:
            transformed_image = op(transformed_image)
        processed_data.append((transformed_image, boxes, labels))
        print(f"  Processed {i+1}/{len(coco_ds)} images...", end='\r')
    
    end_time = time.time()
    print(f"\nDataset pre-processing finished in {end_time - start_time:.2f} seconds.")

    # 2. Use the fast in-memory generator
    dataset = GeneratorDataset(
        source=lambda: create_preloaded_generator(processed_data, BATCH_SIZE),
        column_names=["image", "boxes", "labels"],
        num_parallel_workers=1 # Generator is already fast, no need for more workers
    )
    
    print("Fast in-memory dataset pipeline created.")

    print("\nCreating model, loss function, and optimizer...")
    model = TrafficVisionDetector(num_classes=8)
    loss_fn = DetectionLoss()
    network_with_loss = WithLossCell(model, loss_fn)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.0001)

    trainer = train.Model(network=network_with_loss, optimizer=optimizer)

    loss_monitor = train.LossMonitor(per_print_times=16)
    checkpoint_cb = train.ModelCheckpoint(prefix="traffic_vision_full", directory="./checkpoints")
    step_monitor = StepAndEpochMonitor()

    print("\n--- Starting Full Training ---")
    trainer.train(
        epoch=20,
        train_dataset=dataset,
        callbacks=[loss_monitor, checkpoint_cb, step_monitor]
    )

    print("\n--- Training Finished ---")