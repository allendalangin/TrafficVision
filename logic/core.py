# /logic/core.py

import flet as ft
import datetime
import random
from typing import List, Optional, Union
import httpx
import asyncio
import sqlite3
import json
import os
import shutil
from fpdf import FPDF 

# Import Models from the root level
from data_models import AnalysisResult, BatchAnalysis, Classification, ClassificationResult

# --- Global Config (Kept here for simple setup) ---
PERSISTENT_IMAGE_DIR = "persistent_images"


# --- Helper Class: UserSettings (Renamed/Moved) ---
class UserSettings:
    """Corresponds to the UserSettings class in the diagram."""
    def __init__(self, confidence_threshold: float = 70.0):
        self.confidence_threshold = confidence_threshold

# --- Database Manager ---

class DatabaseManager:
    """Handles all SQLite database operations."""
    def __init__(self, db_path: str = "traffic_vision.db"):
        self.db_path = db_path
        self._create_tables()

    def _get_conn(self):
        # We need a custom row factory to access columns by name (e.g., row['analysis_id'])
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        """Creates database tables if they don't exist."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    image_name TEXT,
                    processing_time REAL,
                    confidence_threshold REAL,
                    batch_id TEXT 
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT,
                    class_name TEXT,
                    confidence REAL,
                    FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def save_analysis(self, analysis: AnalysisResult, batch_id: Optional[str] = None):
        """Saves a single analysis result to the database."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO analyses (analysis_id, timestamp, image_name, processing_time, confidence_threshold, batch_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        analysis.analysis_id,
                        analysis.timestamp.isoformat(),
                        analysis.image_name, 
                        analysis.results.processing_time,
                        analysis.results.confidence_threshold,
                        batch_id
                    ),
                )
                
                class_data = [
                    (analysis.analysis_id, c.class_name, c.confidence)
                    for c in analysis.results.classifications
                ]
                cursor.executemany(
                    "INSERT INTO classifications (analysis_id, class_name, confidence) VALUES (?, ?, ?)",
                    class_data,
                )
                conn.commit()
                # print(f"--- DEBUG: Successfully saved analysis {analysis.analysis_id} to DB ---")
            except sqlite3.IntegrityError:
                pass # print(f"--- DEBUG: Analysis {analysis.analysis_id} already exists in DB. Skipping. ---")
            except Exception as e:
                print(f"--- DEBUG: Error saving to DB: {e} ---")
                conn.rollback()

    def _load_classifications(self, conn: sqlite3.Connection) -> dict:
        """Helper to load all classifications into a map."""
        class_map = {}
        cursor = conn.cursor()
        # Ensure row factory is set to support column names if you use the _get_conn method
        cursor.execute("SELECT analysis_id, class_name, confidence FROM classifications")
        for row in cursor.fetchall():
            analysis_id = row["analysis_id"]
            if analysis_id not in class_map:
                class_map[analysis_id] = []
            class_map[analysis_id].append(
                Classification(className=row["class_name"], confidence=row["confidence"])
            )
        return class_map

    def _build_analysis_from_row(self, row: sqlite3.Row, class_map: dict) -> AnalysisResult:
        """Helper to build an AnalysisResult object from a DB row."""
        analysis_id = row["analysis_id"]
        classifications = class_map.get(analysis_id, [])
        return AnalysisResult(
            analysisId=analysis_id,
            timestamp=datetime.datetime.fromisoformat(row["timestamp"]),
            imageName=row["image_name"],
            results=ClassificationResult(
                classifications=classifications,
                processingTime=row["processing_time"],
                confidenceThreshold=row["confidence_threshold"],
            ),
        )

    def load_history(self) -> List[Union[AnalysisResult, BatchAnalysis]]:
        """Loads all analyses from the database, grouping them into batches."""
        final_list = []
        batch_map = {}
        
        with self._get_conn() as conn:
            class_map = self._load_classifications(conn)
            
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses ORDER BY timestamp DESC")
            
            for row in cursor.fetchall():
                try:
                    analysis = self._build_analysis_from_row(row, class_map)
                    batch_id = row["batch_id"]
                    
                    if batch_id:
                        if batch_id not in batch_map:
                            batch_map[batch_id] = BatchAnalysis(
                                batch_id=batch_id,
                                timestamp=analysis.timestamp, 
                                analyses=[]
                            )
                        batch_map[batch_id].analyses.append(analysis)
                    else:
                        final_list.append(analysis)
                except Exception as e:
                    print(f"--- DEBUG: Error parsing row from DB: {e} ---")
            
        final_list.extend(batch_map.values())
        final_list.sort(key=lambda x: x.timestamp, reverse=True)
            
        return final_list
        
    def load_analysis_by_id(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Loads a single analysis by its ID."""
        with self._get_conn() as conn:
            class_map = self._load_classifications(conn)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,))
            row = cursor.fetchone()
            if row:
                return self._build_analysis_from_row(row, class_map)
        return None

    def load_batch_by_id(self, batch_id: str) -> Optional[BatchAnalysis]:
        """Loads a batch of analyses by its ID."""
        with self._get_conn() as conn:
            class_map = self._load_classifications(conn)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses WHERE batch_id = ? ORDER BY timestamp DESC", (batch_id,))
            
            analyses = []
            for row in cursor.fetchall():
                analyses.append(self._build_analysis_from_row(row, class_map))
            
            if analyses:
                return BatchAnalysis(
                    batch_id=batch_id,
                    timestamp=analyses[0].timestamp, 
                    analyses=analyses
                )
        return None

# --- State Management (Session) ---

class Session:
    """Manages application state, settings, and historical data retrieval."""
    def __init__(self):
        self.session_id = f"sess_{random.randint(1000, 9999)}"
        self.start_time = datetime.datetime.now()
        self.user_settings = UserSettings(confidence_threshold=50.0)
        
        self.db = DatabaseManager()
        self.analyses: List[Union[AnalysisResult, BatchAnalysis]] = self.db.load_history()

    def get_history(self) -> List[Union[AnalysisResult, BatchAnalysis]]:
        """Returns the in-memory list of analyses, sorted."""
        self.analyses.sort(key=lambda x: x.timestamp, reverse=True)
        return self.analyses

    def add_analysis(self, analysis: AnalysisResult, batch_id: Optional[str] = None):
        """Adds an analysis to in-memory list and saves to DB."""
        if batch_id:
            found_batch = next((item for item in self.analyses if isinstance(item, BatchAnalysis) and item.batch_id == batch_id), None)
            if found_batch:
                if not any(a.analysis_id == analysis.analysis_id for a in found_batch.analyses):
                    found_batch.analyses.append(analysis)
                    found_batch.timestamp = max(a.timestamp for a in found_batch.analyses)
            else:
                new_batch = BatchAnalysis(batch_id=batch_id, timestamp=analysis.timestamp, analyses=[analysis])
                self.analyses.append(new_batch)
        else:
            if not any(isinstance(item, AnalysisResult) and item.analysis_id == analysis.analysis_id for item in self.analyses):
                self.analyses.append(analysis)
            
        self.db.save_analysis(analysis, batch_id)
        
    def get_user_settings(self) -> UserSettings:
        return self.user_settings
    
    def remove_analysis(self, analysis_id: str):
        """Removes a single analysis from the in-memory list."""
        
        # 1. Check for single analysis removal
        self.analyses = [
            item for item in self.analyses 
            if not (isinstance(item, AnalysisResult) and item.analysis_id == analysis_id)
        ]

        # 2. Check for removal from a batch
        for batch in [item for item in self.analyses if isinstance(item, BatchAnalysis)]:
            original_count = len(batch.analyses)
            batch.analyses = [
                analysis for analysis in batch.analyses 
                if analysis.analysis_id != analysis_id
            ]
            # If the batch is now empty, remove the batch itself
            if not batch.analyses:
                self.analyses.remove(batch)
            # If removed from a batch, we're done
            if len(batch.analyses) < original_count:
                return

# --- API Client ---

class APIClient:
    """A real API client to talk to the FastAPI server."""
    def __init__(self, session: Session, base_url: str):
        self.session = session
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def upload_image_by_path(self, file_path: str, file_name: str, batch_id: Optional[str] = None) -> Optional[AnalysisResult]:
        """Uploads a single image from a path."""
        
        threshold_pct = self.session.get_user_settings().confidence_threshold
        threshold_float = round(threshold_pct / 100.0, 2)
        
        try:
            # Persistent copy logic
            unique_filename = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}_{file_name}"
            persistent_path = os.path.join(PERSISTENT_IMAGE_DIR, unique_filename)
            shutil.copy(file_path, persistent_path)
            
            with open(persistent_path, "rb") as f: 
                files_payload = {'file': (file_name, f, 'image/jpeg')} 
                params = {'confidence_threshold': threshold_float}
                
                response = await self.client.post("/analyze", files=files_payload, params=params)

            if response.status_code == 200:
                response_json = response.json()
                
                analysis_result = AnalysisResult.model_validate(response_json)
                analysis_result.image_name = persistent_path 
                
                self.session.add_analysis(analysis_result, batch_id=batch_id) 
                return analysis_result 
                
            else:
                detail = response.json().get('detail', 'Unknown server error')
                raise Exception(f"API Error {response.status_code}: {detail}")

        except httpx.ConnectError:
            raise Exception("Connection Error: Is the API server running?")
        except Exception as e:
            raise e 

    async def upload_image(self, file: Optional[ft.FilePickerResultEvent]) -> Optional[AnalysisResult]:
        """Uploads a single image from a FilePicker event."""
        if not file or not file.files:
            return None 

        file_path = file.files[0].path
        file_name = file.files[0].name
        
        return await self.upload_image_by_path(file_path, file_name, batch_id=None) 
    
    async def delete_analysis_by_id(self, analysis_id: str):
        """Sends a DELETE request to the backend."""
        try:
            response = await self.client.delete(f"/analysis/{analysis_id}")
            if response.status_code == 200:
                print(f"--- API SUCCESS: Deleted analysis {analysis_id} ---")
                return True
            else:
                detail = response.json().get('detail', 'Unknown server error')
                raise Exception(f"API Error {response.status_code}: {detail}")
        except httpx.ConnectError:
            raise Exception("Connection Error: Is the API server running?")
        except Exception as e:
            raise e


# --- Export Manager (Logic/Utility) ---

class ExportManager:
    """Handles exporting analysis results to different formats."""
    
    def __init__(self, save_dialog: ft.FilePicker):
        self.save_dialog = save_dialog

    # --- Public methods that check type ---
    
    def export_pdf(self, page: ft.Page, item: Union[AnalysisResult, BatchAnalysis]):
        if isinstance(item, AnalysisResult):
            self._export_pdf_single(page, item)
        elif isinstance(item, BatchAnalysis):
            self._export_pdf_batch(page, item)

    def export_json(self, item: Union[AnalysisResult, BatchAnalysis]):
        if isinstance(item, AnalysisResult):
            self._export_json_single(item)
        elif isinstance(item, BatchAnalysis):
            self._export_json_batch(item)

    def export_csv(self, item: Union[AnalysisResult, BatchAnalysis]):
        if isinstance(item, AnalysisResult):
            self._export_csv_single(item)
        elif isinstance(item, BatchAnalysis):
            self._export_csv_batch(item)

    # --- Private implementation methods ---

    def _export_pdf_single(self, page: ft.Page, analysis: AnalysisResult):
        """Exports a single analysis to PDF."""
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # ... (PDF generation logic - unchanged) ...
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "TrafficVision Analysis Report", 0, 1, "C")
            pdf.ln(10)
            
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, f"Image: {os.path.basename(analysis.image_name)}", 0, 1)
            pdf.cell(0, 8, f"Analysis ID: {analysis.analysis_id}", 0, 1)
            pdf.cell(0, 8, f"Timestamp: {analysis.timestamp.isoformat()}", 0, 1)
            pdf.ln(5)
            
            pdf.set_font("Arial", "B", 12)
            pdf.cell(100, 10, "Classification", 1)
            pdf.cell(0, 10, "Confidence", 1, 1)
            
            pdf.set_font("Arial", "", 12)
            if not analysis.results.classifications:
                pdf.cell(0, 10, "No objects found.", 1, 1, "C")
            else:
                for c in analysis.results.classifications:
                    pdf.cell(100, 10, f" {c.class_name}", 1)
                    pdf.cell(0, 10, f" {c.confidence*100:.2f}%", 1, 1)
            
            pdf_data = pdf.output()
            
            self.save_dialog.file_name = f"{os.path.basename(analysis.image_name)}.pdf"
            self.save_dialog.file_type = ft.FilePickerFileType.CUSTOM
            self.save_dialog.allowed_extensions = ["pdf"]
            self.save_dialog.data_to_save = pdf_data 
            self.save_dialog.save_mode = "binary"
            self.save_dialog.save_file(dialog_title="Save PDF As")
            
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Error generating PDF: {e}"), bgcolor=ft.Colors.RED_500)
            page.snack_bar.open = True
            page.update()
            
    def _export_pdf_batch(self, page: ft.Page, batch: BatchAnalysis):
        """Exports a full batch analysis to PDF."""
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # ... (PDF generation logic - unchanged) ...
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "TrafficVision Batch Analysis Report", 0, 1, "C")
            pdf.ln(5)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, f"Batch ID: {batch.batch_id}", 0, 1)
            pdf.cell(0, 8, f"Timestamp: {batch.timestamp.isoformat()}", 0, 1)
            pdf.cell(0, 8, f"Total Images: {len(batch.analyses)}", 0, 1)
            pdf.ln(10)

            for analysis in batch.analyses:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, f"Image: {os.path.basename(analysis.image_name)}", 0, 1)
                
                pdf.set_font("Arial", "B", 10)
                pdf.cell(100, 8, "Classification", 1)
                pdf.cell(0, 8, "Confidence", 1, 1)
                
                pdf.set_font("Arial", "", 10)
                if not analysis.results.classifications:
                    pdf.cell(0, 8, "No objects found.", 1, 1, "C")
                else:
                    for c in analysis.results.classifications:
                        pdf.cell(100, 8, f" {c.class_name}", 1)
                        pdf.cell(0, 8, f" {c.confidence*100:.2f}%", 1, 1)
                pdf.ln(5)

            pdf_data = pdf.output()
            
            self.save_dialog.file_name = f"{batch.batch_id}.pdf"
            self.save_dialog.file_type = ft.FilePickerFileType.CUSTOM
            self.save_dialog.allowed_extensions = ["pdf"]
            self.save_dialog.data_to_save = pdf_data
            self.save_dialog.save_mode = "binary"
            self.save_dialog.save_file(dialog_title="Save Batch PDF As")
            
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Error generating PDF: {e}"), bgcolor=ft.Colors.RED_500)
            page.snack_bar.open = True
            page.update()
            
    def _export_json_single(self, analysis: AnalysisResult):
        """Exports a single analysis to JSON."""
        json_data = analysis.model_dump_json(indent=4)
        
        self.save_dialog.file_name = f"{os.path.basename(analysis.image_name)}.json"
        self.save_dialog.file_type = ft.FilePickerFileType.CUSTOM
        self.save_dialog.allowed_extensions = ["json"]
        self.save_dialog.data_to_save = json_data
        self.save_dialog.save_mode = "text"
        self.save_dialog.save_file(dialog_title="Save JSON As")
        
    def _export_json_batch(self, batch: BatchAnalysis):
        """Exports a full batch analysis to JSON."""
        json_data = batch.model_dump_json(indent=4)
        
        self.save_dialog.file_name = f"{batch.batch_id}.json"
        self.save_dialog.file_type = ft.FilePickerFileType.CUSTOM
        self.save_dialog.allowed_extensions = ["json"]
        self.save_dialog.data_to_save = json_data
        self.save_dialog.save_mode = "text"
        self.save_dialog.save_file(dialog_title="Save Batch JSON As")


    def _export_csv_single(self, analysis: AnalysisResult):
        """Exports a single analysis to CSV."""
        img_name = os.path.basename(analysis.image_name)
        csv_header = "image_name,class_name,confidence\n"
        csv_rows = [
            f"{img_name},{c.class_name},{c.confidence}"
            for c in analysis.results.classifications
        ]
        csv_data = csv_header + "\n".join(csv_rows)
        
        self.save_dialog.file_name = f"{img_name}.csv"
        self.save_dialog.file_type = ft.FilePickerFileType.CUSTOM
        self.save_dialog.allowed_extensions = ["csv"]
        self.save_dialog.data_to_save = csv_data
        self.save_dialog.save_mode = "text"
        self.save_dialog.save_file(dialog_title="Save CSV As")

    def _export_csv_batch(self, batch: BatchAnalysis):
        """Exports a full batch analysis to CSV."""
        
        csv_header = "image_name,class_name,confidence\n"
        csv_rows = []
        for analysis in batch.analyses:
            img_name = os.path.basename(analysis.image_name)
            if not analysis.results.classifications:
                csv_rows.append(f"{img_name},None,0.0")
            else:
                for c in analysis.results.classifications:
                    csv_rows.append(f"{img_name},{c.class_name},{c.confidence}")
        
        csv_data = csv_header + "\n".join(csv_rows)
        
        self.save_dialog.file_name = f"{batch.batch_id}.csv"
        self.save_dialog.file_type = ft.FilePickerFileType.CUSTOM
        self.save_dialog.allowed_extensions = ["csv"]
        self.save_dialog.data_to_save = csv_data
        self.save_dialog.save_mode = "text"
        self.save_dialog.save_file(dialog_title="Save Batch CSV As")