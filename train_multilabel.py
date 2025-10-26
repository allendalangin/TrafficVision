import mindspore as ms
from mindcv.models import create_model
from mindcv.optim import create_optimizer
from mindspore import context, ops, nn
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.vision import (
    Resize, Normalize, HWC2CHW, Inter, Rescale  # <-- FIXED: Added Rescale, Removed Decode
)
from mindspore.dataset.transforms import TypeCast
import mindspore.common.dtype as mstype
import numpy as np
import os
import cv2  # We use OpenCV to load images
import json
import time

# --- 1. Custom Dataset Class (FIXED) ---
class CocoMultiLabelDataset:
    """
    Custom dataset for multi-label classification.
    Reads the full COCO images and uses the JSON to create a multi-hot label vector.
    """
    # --- NEW: Added 'split' argument to handle 'train2017' vs 'val2017' ---
    def __init__(self, coco_root, ann_file, classes, split='train'):
        self.img_root = os.path.join(coco_root, f"{split}2017") # <-- NEW
        self.ann_file = os.path.join(coco_root, "annotations", ann_file)
        self.classes = classes
        self.num_classes = len(classes)
        
        print(f"Loading {split} annotations from: {self.ann_file}")
        with open(self.ann_file, 'r') as f:
            data = json.load(f)
            
        self.img_list = data['images']
        
        self.img_to_cats = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            self.img_to_cats.setdefault(img_id, set()).add(cat_id)
            
        print(f"Loaded {len(self.img_list)} images and {len(data['annotations'])} annotations.")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_info = self.img_list[index]
        img_path = os.path.join(self.img_root, img_info['file_name'])
        
        image = cv2.imread(img_path)
        # Handle cases where an image file is corrupted or missing
        if image is None:
            print(f"Warning: Could not read image {img_path}. Returning a black image.")
            image = np.zeros((240, 240, 3), dtype=np.uint8) # Return a placeholder
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        img_id = img_info['id']
        cat_ids = self.img_to_cats.get(img_id, set())
        
        label_vector = np.zeros(self.num_classes, dtype=np.float32)
        for cat_id in cat_ids:
            if cat_id < self.num_classes:
                label_vector[cat_id] = 1.0
                
        return image, label_vector

# --- 2. Setup ---
ms.set_seed(42); np.random.seed(42)
device = "GPU" if "GPU" in (context.get_context("device_target") or "CPU") else "CPU"
context.set_context(mode=ms.GRAPH_MODE, device_target=device)
print("MindSpore device target:", context.get_context("device_target"))

# --- 3. Data Paths and Class Definition (FIXED) ---
coco_dir = "./coco2017" 
# --- NEW: Separate train and val files ---
json_file_train = "instances_train2017_8class_12000.json" 
json_file_val = "instances_val2017_8class.json"

classes = [
    'person', 'bicycle', 'car', 'motorcycle', 
    'bus', 'truck', 'traffic light', 'stop sign'
]
num_classes = len(classes)
print("Classes:", classes, "| num_classes:", num_classes)

# --- 4. Transforms and Data Augmentation (FIXED) ---
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
batch_size = 8
num_workers = 4
image_size = 240

# --- FIXED: Added Rescale(1/255.0, 0.0) to scale pixels to [0, 1] ---
transforms_train = [
    Resize((image_size, image_size), interpolation=Inter.BICUBIC),
    # You can add more augmentations here (e.g., RandomHorizontalFlip)
    Rescale(1/255.0, 0.0), # <-- THIS IS THE CRITICAL FIX
    Normalize(mean=mean, std=std),
    HWC2CHW(),
]

# --- NEW: Validation transforms (no augmentations) ---
transforms_val = [
    Resize((image_size, image_size), interpolation=Inter.BICUBIC),
    Rescale(1/255.0, 0.0), # <-- THIS IS THE CRITICAL FIX
    Normalize(mean=mean, std=std),
    HWC2CHW(),
]

# --- 5. Create Datasets (FIXED) ---
dataset_train_generator = CocoMultiLabelDataset(coco_dir, json_file_train, classes, split='train')
dataset_train = GeneratorDataset(
    source=dataset_train_generator,
    column_names=["image", "label"],
    shuffle=True
)
dataset_train = dataset_train.map(operations=transforms_train, input_columns="image",
                                  num_parallel_workers=num_workers)
dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
print(f"✅ Loaded TRAIN dataset: {dataset_train.get_dataset_size()} batches.")

# --- NEW: Create Validation Dataset ---
dataset_val_generator = CocoMultiLabelDataset(coco_dir, json_file_val, classes, split='val')
dataset_val = GeneratorDataset(
    source=dataset_val_generator,
    column_names=["image", "label"],
    shuffle=False # No shuffle for validation
)
dataset_val = dataset_val.map(operations=transforms_val, input_columns="image",
                                  num_parallel_workers=num_workers)
dataset_val = dataset_val.batch(batch_size, drop_remainder=False) # No drop_remainder
print(f"✅ Loaded VAL dataset: {dataset_val.get_dataset_size()} batches.")


# --- 6. Model, Loss, and Optimizer ---
model = create_model(model_name="efficientnet_b1", num_classes=num_classes, pretrained=True)
model.set_train(True)
print(f"✅ Model: EfficientNet-B1 with {num_classes} classes")

loss_fn = nn.BCEWithLogitsLoss()
optimizer = create_optimizer(model.trainable_params(),
                             opt="adamw", lr=1e-4, weight_decay=1e-4)

# --- 7. Training & Validation Functions ---

def train_one_epoch(model, dataset, loss_fn, optimizer):
    model.set_train(True)
    total_loss = ms.Tensor(0.0, ms.float32)
    total_samples = 0
    total_correct_labels = ms.Tensor(0, ms.int32)
    total_possible_labels = ms.Tensor(0, ms.int32)
    dataset_size = dataset.get_dataset_size() 

    def forward_fn(x, y):
        logits = model(x)
        loss = loss_fn(logits, y.astype(ms.float32))
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    for i, batch in enumerate(dataset.create_dict_iterator()):
        images, labels = batch["image"], batch["label"]
        (loss, logits), grads = grad_fn(images, labels)
        optimizer(grads)
        
        if (i + 1) % 10 == 0 or (i + 1) == dataset_size:
            current_loss = loss.asnumpy()
            print(f"  > Train Step {i+1}/{dataset_size}, Current Loss: {current_loss:.4f}")

        preds = ops.Sigmoid()(logits) > 0.5
        correct = ops.Equal()(preds, labels.bool())
        
        total_correct_labels += correct.astype(ms.int32).sum()
        total_possible_labels += labels.shape[0] * labels.shape[1]
        total_loss += loss
        total_samples += labels.shape[0]

    avg_loss = float((total_loss / total_samples).asnumpy())
    acc = float((total_correct_labels.astype(ms.float32) / total_possible_labels.astype(ms.float32)).asnumpy())
    return avg_loss, acc

# --- NEW: Validation Function ---
def validate(model, dataset, loss_fn):
    model.set_train(False) # Set to evaluation mode
    total_loss = ms.Tensor(0.0, ms.float32)
    total_samples = 0
    total_correct_labels = ms.Tensor(0, ms.int32)
    total_possible_labels = ms.Tensor(0, ms.int32)
    
    for batch in dataset.create_dict_iterator():
        images, labels = batch["image"], batch["label"]
        
        logits = model(images)
        loss = loss_fn(logits, labels.astype(ms.float32))

        preds = ops.Sigmoid()(logits) > 0.5
        correct = ops.Equal()(preds, labels.bool())
        
        total_correct_labels += correct.astype(ms.int32).sum()
        total_possible_labels += labels.shape[0] * labels.shape[1]
        total_loss += loss
        total_samples += labels.shape[0]

    avg_loss = float((total_loss / total_samples).asnumpy())
    acc = float((total_correct_labels.astype(ms.float32) / total_possible_labels.astype(ms.float32)).asnumpy())
    return avg_loss, acc

# --- 8. Training Loop (FIXED) ---
num_epochs = 30
save_dir = "./models"; os.makedirs(save_dir, exist_ok=True)
# --- NEW: Changed path to save the BEST model ---
best_ckpt_path = os.path.join(save_dir, "efficientnet_b1_8class_multilabel_BEST.ckpt")
best_val_acc = 0.0 # Tracker for the best accuracy

print(f"🚀 Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    t0 = time.time()
    
    # Run training
    train_loss, train_acc = train_one_epoch(model, dataset_train, loss_fn, optimizer)
    
    # --- NEW: Run Validation ---
    val_loss, val_acc = validate(model, dataset_val, loss_fn)
    
    print("-" * 50)
    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"{time.time()-t0:.1f}s")
    
    # --- NEW: Save Best Model ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"  ✨ New best val_acc: {best_val_acc:.4f}. Saving checkpoint...")
        ms.save_checkpoint(model, best_ckpt_path)
    print("-" * 50)

print(f"\n✅ Training complete.")
print(f"Best model saved to {best_ckpt_path} (Val Acc: {best_val_acc:.4f})")