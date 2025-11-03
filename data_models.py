# /data_models.py

import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, Field

# --- Data Classes ---

class Classification(BaseModel):
    class_name: str = Field(..., alias="className")
    confidence: float

class ClassificationResult(BaseModel):
    classifications: List[Classification]
    processing_time: float = Field(..., alias="processingTime")
    confidence_threshold: Optional[float] = Field(None, alias="confidenceThreshold")

class AnalysisMetadata(BaseModel):
    analysis_id: str = Field(..., alias="analysisId")
    timestamp: datetime.datetime
    image_name: str = Field(..., alias="imageName")

class AnalysisResult(AnalysisMetadata):
    results: ClassificationResult

class BatchAnalysis(BaseModel):
    batch_id: str
    timestamp: datetime.datetime
    analyses: List[AnalysisResult]