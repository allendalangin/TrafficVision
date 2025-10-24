import mindspore as ms
from mindcv.models import create_model
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindspore import context, ops
from mindspore.dataset import ImageFolderDataset
# NEW: extra vision ops for stronger augmentation + rescaling
from mindspore.dataset.vision import (
    Decode, Resize, Normalize, HWC2CHW, Rescale,
    RandomHorizontalFlip, RandomErasing, Inter)
from mindspore.dataset.transforms import TypeCast
import mindspore.common.dtype as mstype
import numpy as np, os, pathlib, json, time

# --- 1. Setup ---
ms.set_seed(42); np.random.seed(42)
device = "GPU" if "GPU" in (context.get_context("device_target") or "CPU") else "CPU"
context.set_context(mode=ms.GRAPH_MODE, device_target=device)
print("MindSpore device target:", context.get_context("device_target"))

# --- 2. Data Paths and Class Definition ---
train_dir = "../train"
val_dir   = "../val"
assert os.path.isdir(train_dir), f"❌ Train directory not found: {train_dir}"
assert os.path.isdir(val_dir),   f"❌ Validation directory not found: {val_dir}"

# NEW: Use the 8 target classes, sorted for consistency
target_classes = {'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'}
classes = sorted(list(target_classes))
num_classes = len(classes)
print("Classes:", classes, "| num_classes:", num_classes)

# Optional: show counts per class
def count_images(root, cls):
    p = os.path.join(root, cls)
    if not os.path.isdir(p): return 0 # Handle if folder doesn't exist
    return sum(1 for f in os.listdir(p) if f.lower().endswith((".jpg",".jpeg",".png")))

print("Train counts:", {c: count_images(train_dir, c) for c in classes})
print("Val counts:  ", {c: count_images(val_dir, c) for c in classes})

dataset_train = ImageFolderDataset(dataset_dir=train_dir, shuffle=True, decode=False)
dataset_val   = ImageFolderDataset(dataset_dir=val_dir,   shuffle=False, decode=False)
print(f"✅ Loaded dataset: {dataset_train.get_dataset_size()} train, {dataset_val.get_dataset_size()} val samples.")

# --- 3. Transforms and Data Augmentation ---
mean = [0.485, 0.456, 0.406] # ImageNet mean
std  = [0.229, 0.224, 0.225] # ImageNet std
batch_size = 16
num_workers = 4
image_size = 240 # NEW: EfficientNet-B1 standard input size

# TRAIN transforms
transforms_train = [
    Decode(),
    Resize((image_size, image_size), interpolation=Inter.BICUBIC), # NEW size
    RandomHorizontalFlip(prob=0.5),
    Rescale(1/255.0, 0.0),
    Normalize(mean=mean, std=std),
    HWC2CHW(),
    TypeCast(mstype.float32),
    RandomErasing(prob=0.25),
]

# VAL transforms
transforms_val = [
    Decode(),
    Resize((image_size, image_size), interpolation=Inter.BICUBIC), # NEW size
    Rescale(1/255.0, 0.0),
    Normalize(mean=mean, std=std),
    HWC2CHW(),
    TypeCast(mstype.float32),
]

dataset_train = dataset_train.map(operations=transforms_train, input_columns="image",
                                  num_parallel_workers=num_workers)
dataset_train = dataset_train.batch(batch_size, drop_remainder=True)

dataset_val = dataset_val.map(operations=transforms_val, input_columns="image",
                              num_parallel_workers=num_workers)
dataset_val = dataset_val.batch(batch_size, drop_remainder=True)
print("✅ Datasets decoded, transformed, and batched.")

# --- 4. Model Creation ---
# NEW: Load EfficientNet-B1
model = create_model(model_name="efficientnetb1", num_classes=num_classes, pretrained=True)
model.set_train(True)
print(f"✅ Model: EfficientNet-B1 with {num_classes} classes")

# Optional (staged training): split params for head vs. backbone
# This logic works for EfficientNet as well, as its head is also named 'classifier'
head_params = [p for p in model.trainable_params() if "classifier" in p.name]
backbone_params = [p for p in model.trainable_params() if "classifier" not in p.name]

# --- 5. Loss, Optimizer, and Training Functions ---

# ----- Manual label smoothing setup -----
epsilon = 0.1 # smoothing factor
loss_fn = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')

def smooth_labels(labels, num_classes, eps=0.1):
    on  = ms.Tensor(1.0, ms.float32)
    off = ms.Tensor(0.0, ms.float32)
    oh = ops.one_hot(labels, num_classes, on, off).astype(ms.float32)
    return (1.0 - eps) * oh + eps / num_classes

# ----- Training and Validation Functions -----

def train_one_epoch(model, dataset, loss_fn, optimizer):
    model.set_train(True)
    total_loss = ms.Tensor(0.0, ms.float32)
    total_correct = ms.Tensor(0, ms.int32)
    total_samples = ms.Tensor(0, ms.int32)

    def forward_fn(x, y):
        logits = model(x)
        y_smooth = smooth_labels(y, num_classes, epsilon).astype(ms.float32)
        loss = loss_fn(logits, y_smooth)
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    for batch in dataset.create_dict_iterator():
        images, labels = batch["image"], batch["label"]
        (loss, logits), grads = grad_fn(images, labels)
        optimizer(grads)

        preds = ops.Argmax(axis=1)(logits)
        total_correct += ops.ReduceSum()(ops.Equal()(preds, labels).astype(ms.int32))
        total_samples += labels.shape[0]
        total_loss += loss

    avg_loss = float((total_loss / total_samples.astype(ms.float32)).asnumpy())
    acc = float((total_correct.astype(ms.float32) / total_samples.astype(ms.float32)).asnumpy())
    return avg_loss, acc

def validate(model, dataset):
    model.set_train(False)
    total_correct = 0; total_samples = 0
    for batch in dataset.create_dict_iterator():
        logits = model(batch["image"])
        preds = logits.asnumpy().argmax(axis=1)
        labels = batch["label"].asnumpy()
        total_correct += (preds == labels).sum()
        total_samples += labels.shape[0]
    return total_correct / total_samples

# --- 6. Staged Training Loop ---

# Optimizer for Stage 1 (Head only)
lr_head = 1e-3
optimizer = create_optimizer(head_params if head_params else model.trainable_params(),
                             opt="adamw", lr=lr_head, weight_decay=1e-4)

num_epochs_head = 3     # head-only warmup
num_epochs_full = 12    # full fine-tune (total 15)
best_acc, best_epoch = 0.0, -1
save_dir = pathlib.Path("../models"); save_dir.mkdir(parents=True, exist_ok=True)
# NEW: Updated checkpoint name
best_ckpt_path = str(save_dir / "efficientnetb1_objects_best.ckpt")

print(f"🚀 Training head for {num_epochs_head} epochs...")
for epoch in range(num_epochs_head):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, dataset_train, loss_fn, optimizer)
    val_acc = validate(model, dataset_val)
    print(f"[Head {epoch+1}/{num_epochs_head}] "
          f"Loss:{train_loss:.4f} | Train:{train_acc:.4f} | Val:{val_acc:.4f} | {time.time()-t0:.1f}s")
    
    if val_acc > best_acc:
        best_acc, best_epoch = val_acc, epoch+1
        ms.save_checkpoint(model, best_ckpt_path)

# Optimizer for Stage 2 (Full network)
lr_full = 1e-4
optimizer = create_optimizer(model.trainable_params(), opt="adamw", lr=lr_full, weight_decay=1e-4)

print(f"\n🔧 Fine-tuning all layers for {num_epochs_full} epochs...")
for e in range(num_epochs_full):
    epoch = num_epochs_head + e + 1
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, dataset_train, loss_fn, optimizer)
    val_acc = validate(model, dataset_val)
    print(f"[Full {e+1}/{num_epochs_full}] "
          f"Loss:{train_loss:.4f} | Train:{train_acc:.4f} | Val:{val_acc:.4f} | {time.time()-t0:.1f}s")
    
    if val_acc > best_acc:
        best_acc, best_epoch = val_acc, epoch
        ms.save_checkpoint(model, best_ckpt_path)

print(f"\n✅ Training complete. Best Val Acc: {best_acc:.4f} at epoch {best_epoch}")
print(f"Best checkpoint saved to {best_ckpt_path}")

# --- 7. Save Final Model and Labels ---
# NEW: Updated checkpoint name
last_ckpt_path = str(save_dir / "efficientnetb1_objects_last.ckpt")
ms.save_checkpoint(model, last_ckpt_path)

# Save the labels.json
with open(str(save_dir / "labels_objects.json"), "w") as f:
    json.dump(classes, f, indent=2)

print(f"Saved last checkpoint to {last_Gckpt_path}")
print(f"Saved class labels to {save_dir / 'labels_objects.json'}")