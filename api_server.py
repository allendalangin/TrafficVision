# In api_server.py
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
import time
import datetime
import uuid
import sqlite3
import os
import json
from typing import List, Optional

# Import helper functions and models
from src.detector import TrafficVisionDetector
from api_models import DetectionResults, AnalysisMetadata, AnalysisResult
from api_utils import preprocess_image, post_process_predictions

# --- Configuration ---
MODEL_CKPT_PATH = "./checkpoints/traffic_vision_sample-best.ckpt" # <-- UPDATE THIS PATH to your best checkpoint
NUM_CLASSES = 8
DATABASE_FILE = "traffic_vision_history.db"
UPLOAD_FOLDER = "uploads"
EXPORT_FOLDER = "exports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id TEXT PRIMARY KEY,
            timestamp TEXT,
            image_name TEXT,
            processing_time REAL,
            confidence_threshold REAL,
            detections_json TEXT 
        )
    ''')
    conn.commit()
    conn.close()

init_db() # Create table if it doesn't exist

# --- Load MindSpore Model ---
print("Loading TrafficVisionDetector model...")
# Set context BEFORE loading model
ms.set_context(device_target="CPU") # Or "GPU" if you have one configured
api_model = TrafficVisionDetector(num_classes=NUM_CLASSES)
param_dict = load_checkpoint(MODEL_CKPT_PATH)
load_param_into_net(api_model, param_dict)
api_model.set_train(False)
print(f"Model loaded successfully from {MODEL_CKPT_PATH}")

# --- FastAPI App Definition ---
app = FastAPI(title="TrafficVision API")

# --- API Endpoints ---
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0) # Get threshold from query param
):
    """
    Analyzes a single uploaded image for traffic objects. [cite: 172, 630]
    """
    start_time = time.time()
    image_bytes = await file.read()
    image_name = file.filename or f"upload_{uuid.uuid4()}.jpg"

    try:
        input_tensor = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error preprocessing image: {e}")

    try:
        box_preds, class_preds = api_model(input_tensor)
        results: DetectionResults = post_process_predictions(box_preds, class_preds, confidence_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference or post-processing failed: {e}")

    end_time = time.time()
    processing_time = round(end_time - start_time, 3)
    results.processing_time = processing_time # Update with actual time

    # --- Save to DB ---
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now()
    detections_json = results.model_dump_json(exclude={'processing_time', 'confidence_threshold'}) # Use model_dump_json in Pydantic v2

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO analyses (analysis_id, timestamp, image_name, processing_time, confidence_threshold, detections_json) VALUES (?, ?, ?, ?, ?, ?)",
            (analysis_id, timestamp.isoformat(), image_name, processing_time, confidence_threshold, detections_json)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # Log error, but don't fail the request if DB save fails
        print(f"ERROR: Could not save analysis {analysis_id} to database: {e}")

    # Return combined metadata and results
    return AnalysisResult(
        analysisId=analysis_id,
        timestamp=timestamp,
        imageName=image_name,
        results=results
    )

@app.get("/history", response_model=List[AnalysisMetadata])
async def get_history():
    """
    Retrieves a list of past analysis metadata. [cite: 703, 159]
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT analysis_id, timestamp, image_name FROM analyses ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        
        history = [
            AnalysisMetadata(analysisId=row[0], timestamp=datetime.datetime.fromisoformat(row[1]), imageName=row[2])
            for row in rows
        ]
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {e}")

@app.get("/analysis/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_details(analysis_id: str):
    """
    Retrieves the full details and results for a specific analysis ID. [cite: 710, 159]
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT analysis_id, timestamp, image_name, processing_time, confidence_threshold, detections_json FROM analyses WHERE analysis_id = ?",
            (analysis_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Parse the stored JSON results
        results_data = json.loads(row[5])
        # Manually add back processing time and threshold for the model
        results_data['processingTime'] = row[3]
        results_data['confidenceThreshold'] = row[4]
        
        results = DetectionResults.model_validate(results_data) # Use model_validate in Pydantic v2

        return AnalysisResult(
            analysisId=row[0],
            timestamp=datetime.datetime.fromisoformat(row[1]),
            imageName=row[2],
            results=results
        )
    except HTTPException as http_exc:
         raise http_exc # Re-raise FastAPI errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis details: {e}")

# --- TODO: Implement Export Endpoint [cite: 682] ---
@app.get("/export/{analysis_id}")
async def export_results(analysis_id: str, format: str = Query("json", enum=["json", "csv", "pdf"])):
     # 1. Fetch analysis details using get_analysis_details logic
     # 2. Based on 'format', generate the file content (JSON is easiest, CSV needs formatting, PDF needs a library like reportlab)
     # 3. Save the file to EXPORT_FOLDER (optional)
     # 4. Return a FileResponse (for file download) or JSONResponse
     raise HTTPException(status_code=501, detail="Export not implemented yet")

# --- Run the server (for local testing) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)