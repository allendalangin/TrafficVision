# In api_models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

# Matches BoundingBox class [cite: 537]
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

# Matches Detection class [cite: 530]
class Detection(BaseModel):
    class_name: str = Field(..., alias="className") # Use alias for camelCase if needed by Flet
    confidence: float
    bbox: BoundingBox

# Matches DetectionResults class [cite: 524]
class DetectionResults(BaseModel):
    detections: List[Detection]
    processing_time: float = Field(..., alias="processingTime")
    confidence_threshold: Optional[float] = Field(None, alias="confidenceThreshold")

# Matches Analysis class (simplified for API response) [cite: 384]
class AnalysisMetadata(BaseModel):
    analysis_id: str = Field(..., alias="analysisId")
    timestamp: datetime
    image_name: str = Field(..., alias="imageName") # Assuming we use name instead of path
    # Results might be loaded separately for efficiency

class AnalysisResult(AnalysisMetadata):
     results: DetectionResults

# Matches ServerResponse structure (can be implicitly handled by FastAPI) [cite: 491]
# For explicit structure if needed:
class ServerResponse(BaseModel):
    status_code: int = 200
    data: Optional[Any] = None
    error: Optional[str] = None

    def is_success(self):
        return self.status_code >= 200 and self.status_code < 300