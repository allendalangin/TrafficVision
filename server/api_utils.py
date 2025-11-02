# In api_utils.py
import mindspore as ms
from mindspore import ops
import numpy as np
import cv2
from typing import List

# --- CORRECTED IMPORT: Use '.' for relative import within the 'server' package ---
from .api_models import Classification, ClassificationResult

# --- Model Settings (from predict.py) ---
IMAGE_SIZE = 240
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 
    'bus', 'truck', 'traffic light', 'stop sign'
]
NUM_CLASSES = len(CLASSES)


# --- MODIFIED: Preprocessing from predict.py ---
def preprocess_image(image_bytes: bytes) -> ms.Tensor:
    """Decodes image bytes and applies classification transforms."""
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image")

    # 1. Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Resize
    image_resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    
    # 3. Normalize
    image_norm = image_resized.astype(np.float32) / 255.0
    image_norm = (image_norm - MEAN) / STD
    
    # 4. HWC to CHW
    image_chw = image_norm.transpose(2, 0, 1)
    
    # 5. Add batch dim and convert to Tensor
    image_batch = np.expand_dims(image_chw, axis=0)
    return ms.Tensor(image_batch, ms.float32)


# --- MODIFIED: Post-processing from predict.py ---
def post_process_predictions(logits: ms.Tensor, confidence_threshold: float) -> ClassificationResult:
    """
    Converts raw model logits to user-friendly ClassificationResult.
    """
    # 1. Apply Sigmoid to get probabilities
    sigmoid = ops.Sigmoid()
    probabilities = sigmoid(logits)
    
    # 2. Squeeze batch dim and get NumPy array
    probs_np = probabilities.asnumpy().squeeze() 
    
    # 3. Find results above threshold
    classifications = []
    for i, prob in enumerate(probs_np):
        if prob > confidence_threshold:
            class_name = CLASSES[i]
            classifications.append(
                Classification(className=class_name, confidence=float(prob))
            )

    # 4. Wrap in our Pydantic model
    return ClassificationResult(
        classifications=classifications,
        processingTime=0.0, # This will be updated by the server
        confidenceThreshold=confidence_threshold
    )