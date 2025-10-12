import mindspore as ms
from mindspore import nn, ops, train
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms import TypeCast
import mindspore.dataset.vision as vision
import numpy as np

# Import your custom modules
from src.coco_dataset import CocoDataset
from src.detector import TrafficVisionDetector

def create_batched_generator(source_dataset, batch_size, image_transforms, shuffle=True):
    """
    A Python generator that manually shuffles, transforms, batches, and pads the data.
    Includes detailed print statements for monitoring progress.
    """
    print("[Data Generator] Starting...")
    indices = list(range(len(source_dataset)))
    if shuffle:
        print(f"[Data Generator] Shuffling {len(indices)} indices.")
        np.random.shuffle(indices)

    batch_buffer = []
    batch_count = 0
    total_samples = len(indices)

    for i, idx in enumerate(indices):
        print(f"[Data Generator] Processing sample {i+1}/{total_samples} (Index: {idx})...", end='\r')
        image, boxes, labels = source_dataset[idx]
        batch_buffer.append((image, boxes, labels))

        if len(batch_buffer) == batch_size:
            print(f"\n[Data Generator] Batch buffer full. Creating batch #{batch_count + 1}.")
            images_list = [item[0] for item in batch_buffer]
            boxes_list = [item[1] for item in batch_buffer]
            labels_list = [item[2] for item in batch_buffer]

            # 1. Apply transformations
            transformed_images = []
            for img in images_list:
                transformed_img = img
                for transform_op in image_transforms:
                    transformed_img = transform_op(transformed_img)
                transformed_images.append(transformed_img)
            
            # 2. Perform padding
            max_objects = 0
            for b in boxes_list:
                if b.shape[0] > max_objects:
                    max_objects = b.shape[0]
            if max_objects == 0:
                max_objects = 1

            padded_boxes = []
            padded_labels = []
            for i in range(batch_size):
                box_array = boxes_list[i]
                padded_box = np.zeros((max_objects, 4), dtype=np.float32)
                num_boxes = box_array.shape[0]
                if num_boxes > 0:
                    padded_box[:num_boxes, :] = box_array
                padded_boxes.append(padded_box)

                label_array = labels_list[i]
                padded_label = np.full((max_objects,), -1, dtype=np.int32)
                num_labels = label_array.shape[0]
                if num_labels > 0:
                    padded_label[:num_labels] = label_array
                padded_labels.append(padded_label)

            # 3. Stack and yield
            final_images = np.stack(transformed_images)
            final_boxes = np.stack(padded_boxes)
            final_labels = np.stack(padded_labels)
            
            print(f"[Data Generator] Yielding batch #{batch_count + 1} with shape: Images-{final_images.shape}, Boxes-{final_boxes.shape}, Labels-{final_labels.shape}")
            yield (final_images, final_boxes, final_labels)
            
            batch_count += 1
            batch_buffer = []
    print("\n[Data Generator] Finished processing all samples.")

# --- Loss Function and Training Wrapper ---
class WithLossCell(nn.Cell):
    def __init__(self, network, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, image, gt_boxes, gt_labels):
        box_preds, class_preds = self.network(image)
        return self.loss_fn(box_preds, class_preds, gt_boxes, gt_labels)

class DetectionLoss(nn.Cell):
    def __init__(self):
        super().__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def construct(self, box_preds, class_preds, gt_boxes, gt_labels):
        box_preds = box_preds.transpose((0, 2, 3, 1)).reshape((-1, 4))
        class_preds = class_preds.transpose((0, 2, 3, 1)).reshape((-1, 9))
        if gt_boxes.shape[1] > 0:
            first_pred_box = box_preds[0:1, :]
            first_gt_box = gt_boxes[:, 0, :]
            first_pred_class = class_preds[0:1, :]
            first_gt_label = gt_labels[:, 0]
            loss_reg = self.smooth_l1_loss(first_pred_box, first_gt_box)
            loss_cls = self.cross_entropy_loss(first_pred_class, first_gt_label)
            total_loss = loss_reg + loss_cls
        else:
            total_loss = ops.zeros(1, ms.float32)
        return total_loss

# --- Custom Callback for Step-by-Step Monitoring ---
class StepAndEpochMonitor(train.Callback):
    """A custom callback to monitor training progress at each step and epoch."""
    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        print(f"\n--- Epoch #{cb_params.cur_epoch_num} Begins ---")

    def step_begin(self, run_context):
        cb_params = run_context.original_args()
        print(f"  [Trainer] Step #{cb_params.cur_step_num} Begins...")
        
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        print(f"--- Epoch #{cb_params.cur_epoch_num} Finished ---")

if __name__ == '__main__':
    ms.set_device("CPU")

    print("Initializing COCO dataset...")
    annotations_path = './coco2017/annotations/instances_train2017.json'
    images_path = './coco2017/train2017'
    coco_ds = CocoDataset(annotations_file=annotations_path, images_dir=images_path)

    # Define image transformations as a list of operations
    image_transforms = [
        vision.Resize((224, 224)),
        vision.HWC2CHW(),
        TypeCast(ms.float32)
    ]

    print("\nCreating the custom data generator...")
    batched_generator = create_batched_generator(
        source_dataset=coco_ds,
        batch_size=4,
        image_transforms=image_transforms
    )

    # Create the final GeneratorDataset from our custom generator
    dataset = GeneratorDataset(
        source=batched_generator,
        column_names=["image", "boxes", "labels"],
        num_parallel_workers=1 # Set to 0 if running on Windows
    )
    print("MindSpore dataset created from generator.")

    print("\nCreating model, loss function, and optimizer...")
    model = TrafficVisionDetector(num_classes=8)
    loss_fn = DetectionLoss()
    network_with_loss = WithLossCell(model, loss_fn)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.0001)

    trainer = train.Model(network=network_with_loss, optimizer=optimizer)

    # Define callbacks
    loss_monitor = train.LossMonitor(per_print_times=10)
    checkpoint_cb = train.ModelCheckpoint(prefix="traffic_vision", directory="./checkpoints")
    step_monitor = StepAndEpochMonitor() # Our new custom callback

    print("\n--- Starting Training ---")
    print("The trainer.train() function is now being called.")
    trainer.train(
        epoch=5,
        train_dataset=dataset,
        callbacks=[loss_monitor, checkpoint_cb, step_monitor]
    )

    print("\n--- Training Finished ---")