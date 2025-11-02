import flet as ft
import datetime
import random
from typing import List, Optional, Union
import httpx
import asyncio
from pydantic import BaseModel, Field
import sqlite3
import json
import os
import shutil
from fpdf import FPDF # <-- Make sure you have run 'pip install fpdf2'

# --- Directory for saving images persistently ---
PERSISTENT_IMAGE_DIR = "persistent_images"
os.makedirs(PERSISTENT_IMAGE_DIR, exist_ok=True)


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
    image_name: str = Field(..., alias="imageName") # This will now be the *persistent path*

class AnalysisResult(AnalysisMetadata):
    results: ClassificationResult

# --- NEW: A model to represent a completed batch ---
class BatchAnalysis(BaseModel):
    batch_id: str
    timestamp: datetime.datetime
    analyses: List[AnalysisResult]

# --- UI Control for Batch Grid ---
class BatchImageCard(ft.Column):
    """A card that holds an image, its result, and a remove button."""
    def __init__(self, file_path: str, file_name: str, on_remove=None, is_readonly=False):
        super().__init__()
        
        self.file_path = file_path
        self.file_name = file_name
        
        self.result_text = ft.Text(
            "Ready to analyze..." if not is_readonly else "", 
            size=11, 
            color=ft.Colors.GREY_700,
            width=150,
            text_align=ft.TextAlign.CENTER
        )
        
        remove_button = ft.IconButton(
            icon=ft.Icons.CANCEL,
            icon_color=ft.Colors.WHITE,
            icon_size=18,
            on_click=on_remove,
            bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
            tooltip="Remove image",
            style=ft.ButtonStyle(
                shape=ft.CircleBorder(),
                padding=1,
            ),
            visible=not is_readonly
        )
        
        image_stack = ft.Stack(
            [
                ft.Image(
                    src=self.file_path,
                    width=150,
                    height=150,
                    fit=ft.ImageFit.COVER,
                    border_radius=12
                ),
                ft.Container(
                    content=remove_button,
                    top=5,
                    right=5,
                    alignment=ft.alignment.top_right
                )
            ],
            width=150,
            height=150
        )
        
        self.controls = [image_stack, self.result_text]
        self.spacing = 5
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    def set_result(self, text: str, color: str):
        """Updates the result text and color."""
        self.result_text.value = text
        self.result_text.color = color
        
# --- (Old UserSettings class is fine) ---
class UserSettings:
    """Corresponds to the UserSettings class in the diagram."""
    def __init__(self, confidence_threshold: float = 70.0):
        self.confidence_threshold = confidence_threshold

# --- NEW: Database Manager ---

class DatabaseManager:
    """Handles all SQLite database operations."""
    def __init__(self, db_path: str = "traffic_vision.db"):
        self.db_path = db_path
        self._create_tables()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

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
                print(f"--- DEBUG: Successfully saved analysis {analysis.analysis_id} to DB ---")
            except sqlite3.IntegrityError:
                print(f"--- DEBUG: Analysis {analysis.analysis_id} already exists in DB. Skipping. ---")
            except Exception as e:
                print(f"--- DEBUG: Error saving to DB: {e} ---")
                conn.rollback()

    def _load_classifications(self, conn: sqlite3.Connection) -> dict:
        """Helper to load all classifications into a map."""
        class_map = {}
        cursor = conn.cursor()
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
        print("--- DEBUG: Loading history from database... ---")
        final_list = []
        batch_map = {}
        
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
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
            
        print(f"--- DEBUG: Loaded {len(final_list)} history items ({len(batch_map)} batches) from DB ---")
        return final_list
        
    def load_analysis_by_id(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Loads a single analysis by its ID."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
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
            conn.row_factory = sqlite3.Row
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

# --- State Management (from Class Diagram) ---

class Session:
    """Corresponds to the Session class in the diagram."""
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

# --- NEW: Real APIClient ---

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
            unique_filename = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}_{file_name}"
            persistent_path = os.path.join(PERSISTENT_IMAGE_DIR, unique_filename)
            shutil.copy(file_path, persistent_path)
            print(f"--- DEBUG: Copied file to {persistent_path} ---")
            
            with open(persistent_path, "rb") as f: 
                files_payload = {'file': (file_name, f, 'image/jpeg')} 
                params = {'confidence_threshold': threshold_float}
                
                print(f"\n--- DEBUG: Sending request to /analyze for {file_name} ---")
                response = await self.client.post("/analyze", files=files_payload, params=params)

            if response.status_code == 200:
                response_json = response.json()
                print(f"--- API SUCCESS (Single): {response.status_code} ---")
                
                analysis_result = AnalysisResult.model_validate(response_json)
                analysis_result.image_name = persistent_path 
                
                self.session.add_analysis(analysis_result, batch_id=batch_id) 
                return analysis_result 
                
            else:
                print(f"--- API ERROR (Single): {response.status_code} ---")
                print(f"--- API Response Text: {response.text} ---")
                detail = response.json().get('detail', 'Unknown server error')
                raise Exception(f"{response.status_code} - {detail}")

        except httpx.ConnectError as e:
            print(f"--- DEBUG: httpx.ConnectError: {e} ---")
            raise Exception("Connection Error: Is the API server running?")
        except Exception as e:
            print(f"--- DEBUG: Unexpected Error in upload_image: {e} ---")
            raise e 

    async def upload_image(self, file: Optional[ft.FilePickerResultEvent]) -> Optional[AnalysisResult]:
        """Uploads a single image from a FilePicker event."""
        if not file or not file.files:
            print("--- DEBUG: No file selected in APIClient ---")
            return None 

        file_path = file.files[0].path
        file_name = file.files[0].name
        
        return await self.upload_image_by_path(file_path, file_name, batch_id=None) 


# --- MODIFIED: ExportManager ---

class ExportManager:
    """Handles exporting analysis results to different formats."""
    
    def __init__(self, save_dialog: ft.FilePicker):
        self.save_dialog = save_dialog

    # --- NEW: Public methods that check type ---
    
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
        print(f"--- DEBUG: Generating PDF for {analysis.image_name} ---")
        
        try:
            pdf = FPDF()
            pdf.add_page()
            
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
            print(f"--- DEBUG: Error generating PDF: {e} ---")
            page.snack_bar = ft.SnackBar(ft.Text(f"Error generating PDF: {e}"), bgcolor=ft.Colors.RED_500)
            page.snack_bar.open = True
            page.update()
            
    def _export_pdf_batch(self, page: ft.Page, batch: BatchAnalysis):
        """Exports a full batch analysis to PDF."""
        print(f"--- DEBUG: Generating PDF for batch {batch.batch_id} ---")
        try:
            pdf = FPDF()
            pdf.add_page()
            
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
                pdf.ln(5) # Add space between images

            pdf_data = pdf.output()
            
            self.save_dialog.file_name = f"{batch.batch_id}.pdf"
            self.save_dialog.file_type = ft.FilePickerFileType.CUSTOM
            self.save_dialog.allowed_extensions = ["pdf"]
            self.save_dialog.data_to_save = pdf_data
            self.save_dialog.save_mode = "binary"
            self.save_dialog.save_file(dialog_title="Save Batch PDF As")
            
        except Exception as e:
            print(f"--- DEBUG: Error generating Batch PDF: {e} ---")
            page.snack_bar = ft.SnackBar(ft.Text(f"Error generating PDF: {e}"), bgcolor=ft.Colors.RED_500)
            page.snack_bar.open = True
            page.update()
        
    def _export_json_single(self, analysis: AnalysisResult):
        """Exports a single analysis to JSON."""
        print(f"--- DEBUG: Exporting JSON for {analysis.image_name} ---")
        json_data = analysis.model_dump_json(indent=4)
        
        self.save_dialog.file_name = f"{os.path.basename(analysis.image_name)}.json"
        self.save_dialog.file_type = ft.FilePickerFileType.CUSTOM
        self.save_dialog.allowed_extensions = ["json"]
        self.save_dialog.data_to_save = json_data
        self.save_dialog.save_mode = "text"
        self.save_dialog.save_file(dialog_title="Save JSON As")
        
    def _export_json_batch(self, batch: BatchAnalysis):
        """Exports a full batch analysis to JSON."""
        print(f"--- DEBUG: Exporting JSON for batch {batch.batch_id} ---")
        json_data = batch.model_dump_json(indent=4)
        
        self.save_dialog.file_name = f"{batch.batch_id}.json"
        self.save_dialog.file_type = ft.FilePickerFileType.CUSTOM
        self.save_dialog.allowed_extensions = ["json"]
        self.save_dialog.data_to_save = json_data
        self.save_dialog.save_mode = "text"
        self.save_dialog.save_file(dialog_title="Save Batch JSON As")


    def _export_csv_single(self, analysis: AnalysisResult):
        """Exports a single analysis to CSV."""
        print(f"--- DEBUG: Exporting CSV for {analysis.image_name} ---")
        
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
        print(f"--- DEBUG: Exporting CSV for batch {batch.batch_id} ---")
        
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

# --- UI View Logic Class ---
class AppLogic:
    """Holds application state and view building logic."""
    
    def __init__(self, session: Session, export_manager: ExportManager): 
        print("--- DEBUG: AppLogic __init__() started ---")
        self.session = session 
        self.api_client = APIClient(session=session, base_url="http://127.0.0.1:8000") 
        self.export_manager = export_manager
    
        self.batch_upload_result_text = ft.Text("", color=ft.Colors.GREY_600, size=12, italic=True)
        
        common_text_style = {
            "size": 14,
            "weight": ft.FontWeight.BOLD,
            "text_align": ft.TextAlign.CENTER, 
            "width": 450, 
        }
        self.single_analysis_output_text = ft.Text(value="", **common_text_style)
        self.batch_analysis_output_text = ft.Text(value="", **common_text_style)

        self.single_image_display = ft.Image(fit=ft.ImageFit.CONTAIN, expand=True)
        self.single_image_placeholder = ft.Column(
                [ft.Icon(ft.Icons.IMAGE_NOT_SUPPORTED, size=50, color=ft.Colors.GREY_400), ft.Text("No image selected", color=ft.Colors.GREY_600)],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
        self.single_image_container = ft.Container(
            content=self.single_image_placeholder,
            width=500,
            height=500,
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=12,
            alignment=ft.alignment.center,
            bgcolor=ft.Colors.GREY_50
        )
        
        self.batch_image_grid = ft.GridView(
            expand=True,
            runs_count=3,
            max_extent=200,
            child_aspect_ratio=0.8,
            spacing=10,
            run_spacing=10,
        )
        
        print("--- DEBUG: AppLogic __init__() finished ---")
        
    def on_slider_changed(self, e: ft.ControlEvent):
        """Updates the session and the text when the slider moves."""
        print(f"--- DEBUG: Slider changed to {e.control.value} ---")
        new_value = int(e.control.value)
        self.session.get_user_settings().confidence_threshold = new_value
        
        text_control = e.control.parent.controls[0]
        text_control.value = f"Confidence Threshold: {new_value}%"
        
        if e.control.page:
            e.control.page.update()
        else:
            print("--- DEBUG: Error! e.control.page is None in on_slider_changed ---")

    def on_settings_click(self, e: ft.ControlEvent):
        """Toggles the visibility of the settings panel."""
        print("--- DEBUG: on_settings_click() triggered ---")
        
        stack = e.page.views[-1].controls[0]
        settings_panel = stack.controls[1]
        
        settings_panel.visible = not settings_panel.visible
        e.page.update()
        print(f"--- DEBUG: Settings panel visibility set to: {settings_panel.visible} ---")

    def remove_batch_card(self, e: ft.ControlEvent):
        """Removes a card from the batch grid."""
        card_to_remove = e.control.parent.parent.parent
        
        if isinstance(card_to_remove, BatchImageCard):
            print(f"--- DEBUG: Removing card {card_to_remove.file_name} ---")
            self.batch_image_grid.controls.remove(card_to_remove)
            
            count = len(self.batch_image_grid.controls)
            if count == 0:
                self.batch_upload_result_text.value = "No files selected."
            else:
                self.batch_upload_result_text.value = f"Selected {count} files"
            
            e.page.update()

    def _build_settings_panel(self) -> ft.Container:
        """Helper method to create a new settings panel instance."""
        
        confidence_text = ft.Text(
            f"Confidence Threshold: {int(self.session.get_user_settings().confidence_threshold)}%", 
            weight=ft.FontWeight.BOLD
        )
        
        return ft.Container(
            content=ft.Column(
                [
                    confidence_text, 
                    ft.Slider(
                        min=0, max=100, 
                        value=self.session.get_user_settings().confidence_threshold, 
                        divisions=100, 
                        active_color=ft.Colors.BLACK, 
                        inactive_color=ft.Colors.GREY_300, 
                        on_change=self.on_slider_changed, 
                    ),
                    ft.Text("Minimum confidence level (0-100%)", color=ft.Colors.GREY_600, size=12),
                ],
                tight=True,
            ),
            top=60,
            right=30,
            visible=False, 
            bgcolor=ft.Colors.WHITE,
            padding=15,
            border_radius=10,
            shadow=ft.BoxShadow(blur_radius=15, color=ft.Colors.with_opacity(0.1, ft.Colors.BLACK)),
            border=ft.border.all(1, ft.Colors.GREY_300),
            width=300,
        )

    # --- View Builders ---
    def build_main_menu_view(self, page: ft.Page):
        """Builds the main dashboard/menu view."""
        print("--- DEBUG: 7a. build_main_menu_view() called ---") 
        
        def menu_card(icon, title, desc, color, on_click=None):
             return ft.Container(
                content=ft.Column(
                    [
                        ft.Container(
                            content=ft.Icon(icon, size=40, color=color),
                            bgcolor=ft.Colors.with_opacity(0.15, color),
                            border_radius=12,
                            padding=10,
                        ),
                        ft.Text(title, size=18, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                        ft.Text(desc, size=13, text_align=ft.TextAlign.CENTER, color=ft.Colors.GREY_600),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=8,
                ),
                width=250,
                height=220,
                alignment=ft.alignment.center,
                padding=20,
                bgcolor=ft.Colors.WHITE,
                border_radius=16,
                shadow=ft.BoxShadow(blur_radius=8, color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK)),
                ink=True,
                on_click=on_click,
            )

        menu_cards = ft.Row(
            [
                menu_card(ft.Icons.CAMERA_ALT, "Analyze Image", "Upload a single traffic image...", ft.Colors.BLUE_600, 
                          on_click=lambda e: page.go("/analyze_single_image")), 
                menu_card(ft.Icons.COLLECTIONS, "Analyze Multiple", "Process a batch of images...", ft.Colors.GREEN_600, 
                          on_click=lambda e: page.go("/analyze_multiple_images")), 
                menu_card(ft.Icons.HISTORY, "Check History", "View previous results...", ft.Colors.PURPLE_600, 
                          on_click=lambda e: page.go("/check_history")), 
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=40,
        )
        
        main_content = ft.Column(
            [
                ft.Row( 
                    [
                        ft.Container(
                            content=ft.Row([
                                ft.Container(content=ft.Icon(ft.Icons.PSYCHOLOGY, size=40, color=ft.Colors.BLACK), margin=ft.margin.only(right=10)),
                                ft.Column([
                                    ft.Text("TrafficVision", size=26, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK),
                                    ft.Text("AI-Powered Traffic Image Analysis", size=14, color=ft.Colors.GREY_600),
                                ], spacing=0, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.START),
                            ], alignment=ft.MainAxisAlignment.CENTER),
                            alignment=ft.alignment.center,
                            expand=True
                        ),
                        
                        ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, tooltip="Settings", 
                                      on_click=self.on_settings_click),
                    ], 
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Container(
                    content=menu_cards,
                    expand=True,
                    alignment=ft.alignment.center, 
                ),
            ],
            expand=True, 
            horizontal_alignment=ft.CrossAxisAlignment.CENTER 
        )

        return ft.View(
            "/", 
            [
                ft.Stack(
                    [
                        main_content, 
                        self._build_settings_panel() 
                    ],
                    expand=True
                )
            ]
        )

    def build_analyze_single_image_view(self, page: ft.Page, 
                                        single_file_picker: ft.FilePicker):
        """Builds the 'Analyze Single Image' view with a side-by-side layout."""
        print("--- DEBUG: 7b. build_analyze_single_image_view() called ---")
        
        async def on_analyze_click(e: ft.ControlEvent):
            current_page = e.page 
            file_picker_result = getattr(single_file_picker, 'result', None)
            
            try:
                self.single_analysis_output_text.value = "Analyzing, please wait..."
                self.single_analysis_output_text.color = ft.Colors.BLUE_600
                current_page.update()
                
                analysis_result = await self.api_client.upload_image(file_picker_result)
                
                if analysis_result:
                    class_names = [f"{c.class_name} ({c.confidence*100:.1f}%)" for c in analysis_result.results.classifications]
                    if not class_names:
                        results_str = "No objects found."
                    else:
                        results_str = f"Found: {', '.join(class_names)}"
                    
                    self.single_analysis_output_text.value = results_str
                    self.single_analysis_output_text.color = ft.Colors.BLACK
                else:
                    self.single_analysis_output_text.value = "Please select a file first."
                    self.single_analysis_output_text.color = ft.Colors.RED_500
            
            except Exception as ex:
                print(f"--- DEBUG: Error in on_analyze_click: {ex} ---")
                self.single_analysis_output_text.value = f"Error: {ex}"
                self.single_analysis_output_text.color = ft.Colors.RED_500
            
            current_page.update()

        analyze_button = ft.ElevatedButton(
            "Analyze Image", 
            width=450, 
            bgcolor=ft.Colors.BLUE_600, 
            color=ft.Colors.WHITE,
            on_click=on_analyze_click 
        )

        upload_box = ft.Container(
            content=ft.Column([
                    ft.Icon(ft.Icons.UPLOAD_FILE, size=40, color=ft.Colors.GREY_600),
                    ft.Text("Upload an image", weight=ft.FontWeight.BOLD),
                    ft.Text("PNG, JPG or JPEG", color=ft.Colors.GREY_500, size=12),
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
            border=ft.border.all(1, ft.Colors.GREY_300), border_radius=12, 
            padding=ft.padding.symmetric(vertical=40), 
            width=450, 
            on_click=lambda e: single_file_picker.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE) 
        )

        controls_column = ft.Column(
            [
                ft.Row([
                        ft.Icon(ft.Icons.CAMERA_ALT, color=ft.Colors.BLUE_600), 
                        ft.Text("Analyze Single Image", size=18, weight=ft.FontWeight.BOLD)
                    ], spacing=10),
                ft.Text("Upload a traffic image for AI-powered object classification", 
                        color=ft.Colors.GREY_600, size=13),
                ft.Divider(height=30, color="transparent"), 
                upload_box, 
                ft.Divider(height=20, color="transparent"), 
                analyze_button,
                ft.Divider(height=20, color="transparent"),
                self.single_analysis_output_text,
            ], 
            width=450, 
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER, 
        )
        
        card = ft.Container(
             content=controls_column,
            padding=30, 
            border_radius=20, 
            bgcolor=ft.Colors.WHITE,
            shadow=ft.BoxShadow(blur_radius=12, color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK)),
            alignment=ft.alignment.center_left, 
        )

        main_content = ft.Column(
            [
                ft.Row(
                    [
                        ft.TextButton("← Back to Menu", on_click=lambda e: page.go("/")), 
                        ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, 
                                      on_click=self.on_settings_click), 
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Row( 
                    [
                        card, 
                        ft.Container(width=50), 
                        self.single_image_container, 
                    ], 
                    alignment=ft.MainAxisAlignment.CENTER, 
                    vertical_alignment=ft.CrossAxisAlignment.CENTER, 
                    expand=True,
                ),
            ],
            expand=True
        )

        return ft.View(
            "/analyze_single_image", 
            [
                ft.Stack(
                    [
                        main_content, 
                        self._build_settings_panel() 
                    ],
                    expand=True
                )
            ]
        )

    def build_check_history_view(self, page: ft.Page):
        """Builds the 'Check History' view."""
        print("--- DEBUG: 7c. build_check_history_view() called ---") 
        
        def on_export_pdf(item: Union[AnalysisResult, BatchAnalysis]):
            self.export_manager.export_pdf(page, item)

        def on_export_csv(item: Union[AnalysisResult, BatchAnalysis]):
            self.export_manager.export_csv(item)

        def on_export_json(item: Union[AnalysisResult, BatchAnalysis]):
            self.export_manager.export_json(item)
            
        def build_single_history_card(analysis_item: AnalysisResult):
            avg_conf = 0.0
            class_count = len(analysis_item.results.classifications)
            if class_count > 0:
                avg_conf = sum(c.confidence for c in analysis_item.results.classifications) / class_count
            
            return ft.Container(
                content=ft.Row([
                        ft.Icon(ft.Icons.CAMERA_ALT, color=ft.Colors.BLUE_600),
                        ft.Column([
                                ft.Text(os.path.basename(analysis_item.image_name), weight=ft.FontWeight.BOLD),
                                ft.Text(f"Processed: {analysis_item.timestamp.isoformat()}", color=ft.Colors.GREY_600, size=12),
                                ft.Text(f"Objects found: {class_count} | Avg confidence: {avg_conf*100:.1f}%", color=ft.Colors.GREY_600, size=12),
                            ], expand=True, spacing=2),
                        ft.PopupMenuButton(
                            icon=ft.Icons.MORE_VERT,
                            items=[
                                ft.PopupMenuItem(text="Export as PDF", icon=ft.Icons.PICTURE_AS_PDF, on_click=lambda e, item=analysis_item: on_export_pdf(item)),
                                ft.PopupMenuItem(text="Export as CSV", icon=ft.Icons.TABLE_ROWS, on_click=lambda e, item=analysis_item: on_export_csv(item)),
                                ft.PopupMenuItem(text="Export as JSON", icon=ft.Icons.DATA_OBJECT, on_click=lambda e, item=analysis_item: on_export_json(item)),
                            ]
                        )
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=ft.Colors.WHITE, 
                border_radius=12, 
                padding=15, 
                margin=ft.margin.symmetric(vertical=5),
                shadow=ft.BoxShadow(blur_radius=8, color=ft.Colors.with_opacity(0.15, ft.Colors.BLACK)),
                on_click=lambda e, item=analysis_item: page.go(f"/history/single/{item.analysis_id}")
            )

        def build_batch_history_card(batch_item: BatchAnalysis):
            return ft.Container(
                content=ft.Row([
                        ft.Icon(ft.Icons.COLLECTIONS, color=ft.Colors.GREEN_600),
                        ft.Column([
                                ft.Text("Batch Analysis", weight=ft.FontWeight.BOLD, size=16),
                                ft.Text(f"Processed: {batch_item.timestamp.isoformat()}", color=ft.Colors.GREY_600, size=12),
                                ft.Text(f"{len(batch_item.analyses)} images analyzed", color=ft.Colors.GREY_600, size=12),
                            ], expand=True, spacing=2),
                        # --- NEW: Export menu for batches ---
                        ft.PopupMenuButton(
                            icon=ft.Icons.MORE_VERT,
                            items=[
                                ft.PopupMenuItem(text="Export as PDF", icon=ft.Icons.PICTURE_AS_PDF, on_click=lambda e, item=batch_item: on_export_pdf(item)),
                                ft.PopupMenuItem(text="Export as CSV", icon=ft.Icons.TABLE_ROWS, on_click=lambda e, item=batch_item: on_export_csv(item)),
                                ft.PopupMenuItem(text="Export as JSON", icon=ft.Icons.DATA_OBJECT, on_click=lambda e, item=batch_item: on_export_json(item)),
                            ]
                        )
                        # --- END NEW ---
                    ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=ft.Colors.WHITE, 
                border_radius=12, 
                padding=15, 
                margin=ft.margin.symmetric(vertical=5),
                shadow=ft.BoxShadow(blur_radius=8, color=ft.Colors.with_opacity(0.15, ft.Colors.BLACK)),
                on_click=lambda e, item=batch_item: page.go(f"/history/batch/{item.batch_id}")
            )

        title_section = ft.Column([
                ft.Row([ft.Icon(ft.Icons.HISTORY, color=ft.Colors.PURPLE_600), ft.Text("Analysis History", size=18, weight=ft.FontWeight.BOLD)], spacing=10, alignment=ft.MainAxisAlignment.START),
                ft.Text("View and manage your previous traffic image analyses", color=ft.Colors.GREY_600, size=13),
            ], spacing=4)

        history_list = self.session.get_history()
        
        history_controls = []
        if not history_list:
            history_controls = [ft.Text("No analysis history for this session.", text_align=ft.TextAlign.CENTER, color=ft.Colors.GREY_600, italic=True)]
        else:
            for item in history_list:
                if isinstance(item, AnalysisResult):
                    history_controls.append(build_single_history_card(item))
                elif isinstance(item, BatchAnalysis):
                    history_controls.append(build_batch_history_card(item))
        
        main_content = ft.Column(
            [
                ft.Row([
                        ft.TextButton("← Back to Menu", on_click=lambda e: page.go("/")), 
                        ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, 
                                      on_click=self.on_settings_click), 
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                
                ft.Container(
                    content=ft.Column(
                        [title_section] + history_controls, 
                        spacing=10, 
                        expand=True, 
                        scroll=ft.ScrollMode.ADAPTIVE 
                    ),
                    width=700, 
                    padding=20, 
                    bgcolor=ft.Colors.WHITE, 
                    border_radius=16,
                    shadow=ft.BoxShadow(blur_radius=12, color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK)), 
                    alignment=ft.alignment.top_center,
                    expand=True 
                ),
            ],
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER, 
            alignment=ft.MainAxisAlignment.START
        )
        
        return ft.View(
            "/check_history", 
            [
                ft.Stack(
                    [
                        ft.Container(
                            content=main_content,
                            alignment=ft.alignment.top_center, 
                            expand=True
                        ),
                        self._build_settings_panel()
                    ],
                    expand=True
                )
            ]
        )

    def build_analyze_multiple_images_view(self, page: ft.Page, 
                                           batch_file_picker: ft.FilePicker):
        print("--- DEBUG: 7d. build_analyze_multiple_images_view() called ---")
        
        async def on_analyze_click(e: ft.ControlEvent):
            current_page = e.page
            
            if not self.batch_image_grid.controls:
                self.batch_analysis_output_text.value = "Please select one or more files."
                self.batch_analysis_output_text.color = ft.Colors.RED_500
                current_page.update()
                return

            self.batch_analysis_output_text.value = "Starting batch analysis..."
            self.batch_analysis_output_text.color = ft.Colors.BLUE_600
            current_page.update()

            success_count = 0
            fail_count = 0
            
            batch_id = f"batch_{datetime.datetime.now().isoformat()}"

            for card in self.batch_image_grid.controls:
                if isinstance(card, BatchImageCard):
                    try:
                        card.set_result("Analyzing...", ft.Colors.BLUE_600)
                        current_page.update()

                        result = await self.api_client.upload_image_by_path(
                            card.file_path, card.file_name, batch_id=batch_id
                        )
                        
                        if result:
                            class_names = [f"{c.class_name} ({c.confidence*100:.0f}%)" for c in result.results.classifications]
                            if not class_names:
                                res_str = "No objects"
                            else:
                                res_str = ", ".join(class_names)
                            
                            card.set_result(res_str, ft.Colors.BLACK)
                            success_count += 1
                        else:
                            card.set_result("Analysis failed", ft.Colors.RED_500)
                            fail_count += 1
                    
                    except Exception as ex:
                        print(f"--- DEBUG: Error analyzing {card.file_name}: {ex} ---")
                        card.set_result(f"Error", ft.Colors.RED_500)
                        fail_count += 1
                    
                    current_page.update()

            self.batch_analysis_output_text.value = f"Batch complete: {success_count} succeeded, {fail_count} failed."
            self.batch_analysis_output_text.color = ft.Colors.BLACK if fail_count == 0 else ft.Colors.ORANGE_700
            current_page.update()

        analyze_button = ft.ElevatedButton(
            "Analyze Batch", 
            width=450, 
            bgcolor=ft.Colors.GREEN_600, 
            color=ft.Colors.WHITE,
            on_click=on_analyze_click 
        )

        upload_box = ft.Container(
            content=ft.Column([
                    ft.Icon(ft.Icons.COLLECTIONS, size=40, color=ft.Colors.GREY_600),
                    ft.Text("Upload multiple images", weight=ft.FontWeight.BOLD),
                    ft.Text("PNG, JPG or JPEG files", color=ft.Colors.GREY_500, size=12),
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
            border=ft.border.all(1, ft.Colors.GREY_300), border_radius=12, 
            padding=ft.padding.symmetric(vertical=40), 
            width=450, 
            on_click=lambda e: batch_file_picker.pick_files(allow_multiple=True, file_type=ft.FilePickerFileType.IMAGE) 
        )
        
        controls_column = ft.Column(
            [
                ft.Row([ft.Icon(ft.Icons.COLLECTIONS, color=ft.Colors.GREEN_600), ft.Text("Analyze Multiple Images", size=18, weight=ft.FontWeight.BOLD)], spacing=10),
                ft.Text("Upload a batch of traffic images for classification", 
                        color=ft.Colors.GREY_600, size=13),
                ft.Divider(height=30, color="transparent"),
                upload_box,
                self.batch_upload_result_text, 
                ft.Divider(height=20, color="transparent"),
                analyze_button,
                ft.Divider(height=20, color="transparent"),
                self.batch_analysis_output_text,
            ],
            width=450, 
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER, 
        )
        
        image_grid_container = ft.Container(
            content=self.batch_image_grid,
            width=500,
            height=500,
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=12,
            alignment=ft.alignment.top_left,
            bgcolor=ft.Colors.GREY_50,
            padding=10
        )

        card = ft.Container(
            content=controls_column,
            padding=30, 
            border_radius=20, 
            bgcolor=ft.Colors.WHITE,
            shadow=ft.BoxShadow(blur_radius=12, color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK)), alignment=ft.alignment.center,
        )

        main_content = ft.Column(
            [
                ft.Row([
                        ft.TextButton("← Back to Menu", on_click=lambda e: page.go("/")), 
                        ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, 
                                      on_click=self.on_settings_click), 
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Row(
                    [
                        card,
                        ft.Container(width=50),
                        image_grid_container,
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    expand=True,
                )
            ],
            expand=True
        )
        
        return ft.View(
            "/analyze_multiple_images", 
            [
                ft.Stack(
                    [
                        main_content, 
                        self._build_settings_panel() 
                    ],
                    expand=True
                )
            ]
        )

    # --- NEW: View builder for Single History item ---
    def build_single_history_detail_view(self, page: ft.Page, analysis: AnalysisResult):
        """Shows the details for a single analysis history item."""
        
        results_str = "No objects found."
        if analysis.results.classifications:
            class_names = [f"{c.class_name} ({c.confidence*100:.1f}%)" for c in analysis.results.classifications]
            results_str = f"Found: {', '.join(class_names)}"

        main_content = ft.Column(
            [
                ft.Row(
                    [
                        ft.TextButton("← Back to History", on_click=lambda e: page.go("/check_history")),
                        ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, on_click=self.on_settings_click),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Row(
                    [
                        # Info Card
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text("Analysis Result", size=18, weight=ft.FontWeight.BOLD),
                                    ft.Text(f"Image: {os.path.basename(analysis.image_name)}"),
                                    ft.Text(f"Timestamp: {analysis.timestamp.isoformat()}"),
                                    ft.Divider(height=10, color="transparent"),
                                    ft.Text(results_str, size=16, weight=ft.FontWeight.BOLD),
                                ],
                                spacing=10,
                            ),
                            width=450,
                            padding=30,
                            border_radius=20,
                            bgcolor=ft.Colors.WHITE,
                            shadow=ft.BoxShadow(blur_radius=12, color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK)),
                        ),
                        ft.Container(width=50),
                        # Image
                        ft.Container(
                            content=ft.Image(src=analysis.image_name, fit=ft.ImageFit.CONTAIN, expand=True),
                            width=500,
                            height=500,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=12,
                            alignment=ft.alignment.center,
                            bgcolor=ft.Colors.GREY_50,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    expand=True,
                ),
            ],
            expand=True
        )

        return ft.View(
            f"/history/single/{analysis.analysis_id}",
            [
                ft.Stack(
                    [
                        main_content,
                        self._build_settings_panel()
                    ],
                    expand=True
                )
            ]
        )

    # --- NEW: View builder for Batch History item ---
    def build_batch_history_detail_view(self, page: ft.Page, batch: BatchAnalysis):
        """Shows the details for a batch analysis history item."""

        grid = ft.GridView(
            expand=True,
            runs_count=3,
            max_extent=200,
            child_aspect_ratio=0.8,
            spacing=10,
            run_spacing=10,
        )

        for analysis in batch.analyses:
            class_names = [f"{c.class_name} ({c.confidence*100:.0f}%)" for c in analysis.results.classifications]
            if not class_names:
                res_str = "No objects"
            else:
                res_str = ", ".join(class_names)
            
            card = BatchImageCard(
                file_path=analysis.image_name, # Use persistent path
                file_name=os.path.basename(analysis.image_name),
                is_readonly=True # Hide the 'x' button
            )
            card.set_result(res_str, ft.Colors.BLACK)
            grid.controls.append(card)

        main_content = ft.Column(
            [
                ft.Row(
                    [
                        ft.TextButton("← Back to History", on_click=lambda e: page.go("/check_history")),
                        ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, on_click=self.on_settings_click),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Column(
                    [
                        ft.Text(f"Batch Analysis: {batch.batch_id}", size=20, weight=ft.FontWeight.BOLD),
                        ft.Text(f"Processed: {batch.timestamp.isoformat()} ({len(batch.analyses)} images)"),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER
                ),
                ft.Container(
                    content=grid,
                    expand=True,
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=12,
                    alignment=ft.alignment.top_left,
                    bgcolor=ft.Colors.GREY_50,
                    padding=10
                ),
            ],
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )

        return ft.View(
            f"/history/batch/{batch.batch_id}",
            [
                ft.Stack(
                    [
                        main_content,
                        self._build_settings_panel()
                    ],
                    expand=True
                )
            ]
        )
    
    # --- File Picker Handlers ---
    def on_single_file_result(self, e: ft.FilePickerResultEvent):
        """Handler for when a single file is picked."""
        current_page = e.page
        if not current_page: return
        if e.files:
            print(f"--- DEBUG: Single file picked: {e.files[0].name} ---")
            self.single_image_display.src = e.files[0].path
            self.single_image_container.content = self.single_image_display
            self.single_analysis_output_text.value = "" 
        else:
            print("--- DEBUG: Single file picker cancelled ---")
            self.single_image_container.content = self.single_image_placeholder
        current_page.update()

    def on_batch_file_result(self, e: ft.FilePickerResultEvent):
        """Handler for when multiple files are picked."""
        current_page = e.page
        if not current_page: return
        
        self.batch_image_grid.controls.clear()
        self.batch_analysis_output_text.value = ""
        
        if e.files:
            print(f"--- DEBUG: Batch files picked: {len(e.files)} files ---")
            self.batch_upload_result_text.value = f"Selected {len(e.files)} files"
            
            for file_obj in e.files:
                card = BatchImageCard(
                    file_path=file_obj.path, 
                    file_name=file_obj.name,
                    on_remove=self.remove_batch_card
                )
                self.batch_image_grid.controls.append(card)
        else:
            print("--- DEBUG: Batch file picker cancelled ---")
            self.batch_upload_result_text.value = "No file selected."
        
        current_page.update()


# --- Main App Entry Point ---

async def main(page: ft.Page):
    print("\n" + "="*40)
    print("--- DEBUG: 1. main() started ---") 
    
    page.title = "TrafficVision - AI-Powered Traffic Image Analysis"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30

    # 1. Create the Session and AppLogic
    app_session = Session()
    
    def on_save_file_result(e: ft.FilePickerResultEvent):
        if e.path:
            try:
                data_to_save = e.control.data_to_save
                save_mode = getattr(e.control, "save_mode", "text") 
                
                if save_mode == "binary":
                    with open(e.path, "wb") as f:
                        f.write(data_to_save)
                else:
                    with open(e.path, "w", encoding="utf-8") as f:
                        f.write(data_to_save)
                        
                print(f"--- DEBUG: File saved to {e.path} ---")
                page.snack_bar = ft.SnackBar(ft.Text(f"Successfully exported to {e.path}"), bgcolor=ft.Colors.GREEN_700)
            except Exception as ex:
                print(f"--- DEBUG: Error saving file: {ex} ---")
                page.snack_bar = ft.SnackBar(ft.Text(f"Error saving file: {ex}"), bgcolor=ft.Colors.RED_500)
        else:
            print("--- DEBUG: File save dialog cancelled ---")
            page.snack_bar = ft.SnackBar(ft.Text("Save operation cancelled."))
        
        page.snack_bar.open = True
        page.update()

    save_file_dialog = ft.FilePicker(on_result=on_save_file_result)
    
    export_manager = ExportManager(save_dialog=save_file_dialog)
    app_logic = AppLogic(session=app_session, export_manager=export_manager)
    print("--- DEBUG: 2. AppLogic initialized ---") 

    # 2. Create Dialogs and File Pickers
    
    # -- File Pickers --
    single_file_picker = ft.FilePicker(on_result=app_logic.on_single_file_result)
    batch_file_picker = ft.FilePicker(on_result=app_logic.on_batch_file_result)

    page.overlay.extend([
        single_file_picker, 
        batch_file_picker,
        save_file_dialog 
    ])

    # 3. Define Route Change Handler
    
    async def route_change_handler(e: ft.RouteChangeEvent):
        print(f"\n--- DEBUG: 6. route_change_handler() triggered for route: {e.route} ---") 
        
        current_page = e.page
        current_route = e.route
        current_page.views.clear()
        
        try:
            if current_route == "/":
                view = app_logic.build_main_menu_view(current_page)
            elif current_route == "/analyze_single_image":
                 view = app_logic.build_analyze_single_image_view(current_page, single_file_picker)
            elif current_route == "/analyze_multiple_images":
                 view = app_logic.build_analyze_multiple_images_view(current_page, batch_file_picker)
            elif current_route == "/check_history":
                 view = app_logic.build_check_history_view(current_page)
            
            elif current_route.startswith("/history/single/"):
                analysis_id = current_route.split("/")[-1]
                analysis = app_session.db.load_analysis_by_id(analysis_id)
                if analysis:
                    view = app_logic.build_single_history_detail_view(current_page, analysis)
                else:
                    page.go("/check_history")
                    return
            
            elif current_route.startswith("/history/batch/"):
                batch_id = current_route.split("/")[-1]
                batch = app_session.db.load_batch_by_id(batch_id)
                if batch:
                    view = app_logic.build_batch_history_detail_view(current_page, batch)
                else:
                    page.go("/check_history")
                    return
            
            else:
                print(f"--- DEBUG: Unknown route: {current_route}, redirecting to home. ---") 
                view = app_logic.build_main_menu_view(current_page)
                if current_page.route != "/":
                     current_page.go("/") 
                     return 
            
            print(f"--- DEBUG: 7. View object created for {e.route} ---") 
            current_page.views.append(view)
            
            print("--- DEBUG: 8. Calling page.update() to display view ---") 
            current_page.update()
            print("--- DEBUG: 9. page.update() call complete ---") 
            
        except Exception as ex:
            print(f"!!! --- CRITICAL ERROR in route_change_handler: {ex} --- !!!") 
            import traceback
            traceback.print_exc()

    # 4. Define View Pop Handler
    def view_pop_handler(e: ft.ViewPopEvent):
        print(f"--- DEBUG: view_pop_handler() triggered ---") 
        page.views.pop()
        top_view_route = page.views[-1].route if page.views else "/"
        page.go(top_view_route) 

    # 5. Assign Handlers and Navigate
    page.on_route_change = route_change_handler 
    page.on_view_pop = view_pop_handler
    print("--- DEBUG: 3. Event handlers assigned ---") 
    
    print(f"--- DEBUG: 4. Calling page.go() with initial route: {page.route} ---") 
    page.go(page.route) 
    print("--- DEBUG: 5. Initial page.go() call complete ---") 


if __name__ == "__main__":
    print("--- DEBUG: __main__ block started ---") 
    ft.app(target=main) 
    print("--- DEBUG: ft.app() has exited ---")