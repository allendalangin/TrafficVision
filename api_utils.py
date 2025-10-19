# In api_utils.py
import mindspore as ms
from mindspore import ops
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms import TypeCast
import numpy as np
import cv2
from typing import List, Dict, Any

from api_models import BoundingBox, Detection, DetectionResults # Import Pydantic models

# --- Image Preprocessing ---
def preprocess_image(image_bytes: bytes) -> ms.Tensor:
    """Decodes image bytes and applies necessary transforms."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transforms = [
        vision.Resize((224, 224)),
        vision.HWC2CHW(),
        TypeCast(ms.float32)
    ]
    transformed_image = image
    for op in transforms:
        transformed_image = op(transformed_image)
    return ms.Tensor(np.expand_dims(transformed_image, axis=0))

# --- Prediction Post-processing (CRUCIAL - Needs Proper Implementation) ---
def post_process_predictions(box_preds_ms: ms.Tensor, class_preds_ms: ms.Tensor, confidence_threshold: float = 0.5) -> DetectionResults:
    """
    Converts raw model outputs to user-friendly DetectionResults.
    Includes placeholder logic for NMS and thresholding.
    """
    # Convert MindSpore Tensors to NumPy arrays for processing
    box_preds = box_preds_ms.asnumpy()
    class_preds = class_preds_ms.asnumpy()

    # Get shapes
    batch_size, _, height, width = box_preds.shape
    
    # Reshape predictions (ensure batch dimension is kept)
    # Shape -> (batch_size, H*W*num_anchors, 4) for boxes
    # Shape -> (batch_size, H*W*num_anchors, num_classes + 1) for classes
    num_anchors = 9 # From your SimpleDetectionHead
    num_classes = 8 # Your target classes

    box_preds = box_preds.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 4))
    class_preds = class_preds.transpose((0, 2, 3, 1)).reshape((batch_size, -1, num_classes + 1))

    # --- TODO: Implement Real Post-Processing ---
    # 1. Decode box predictions (e.g., convert offsets back to coordinates).
    # 2. Apply Softmax to class predictions to get probabilities.
    # 3. Filter predictions by confidence threshold.
    # 4. Perform Non-Maximum Suppression (NMS) for each class separately.
    # 5. Map class indices (0-7) back to names ('car', 'person', ...).
    # ----------------------------------------------

    print("WARNING: Using simplified placeholder post-processing.")
    
    detections = []
    # Example: Taking the highest scoring prediction from the first image
    if batch_size > 0:
        scores = class_preds[0].max(axis=1) # Max score across classes for each box
        best_pred_idx = scores.argmax()
        
        best_score = scores[best_pred_idx]
        best_class_idx = class_preds[0, best_pred_idx].argmax()
        best_box = box_preds[0, best_pred_idx]

        if best_score > confidence_threshold and best_class_idx < num_classes: # Ignore background class
             # Map index back to name (you'll need a mapping dictionary)
             class_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']
             class_name = class_names[best_class_idx]
             
             # Create BoundingBox and Detection objects
             # NOTE: Need to convert model's box output format to x1, y1, x2, y2
             # This depends on how your model predicts boxes (center+wh or corners)
             # Placeholder assuming x1,y1,x2,y2 for now - ADJUST THIS!
             x1, y1, x2, y2 = best_box 
             bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
             detections.append(Detection(className=class_name, confidence=float(best_score), bbox=bbox))

    # For now, just return a dummy result or the simple one above
    results = DetectionResults(
        detections=detections,
        processingTime=0.1, # Placeholder time
        confidenceThreshold=confidence_threshold
    )
    return results