import flet as ft
import datetime
import random
from typing import List, Optional

# --- Data Classes (from Class Diagram) ---
# (UserSettings, DetectionResults, Analysis classes remain the same)
class UserSettings:
    """Corresponds to the UserSettings class in the diagram."""
    def __init__(self, confidence_threshold: float = 70.0):
        self.confidence_threshold = confidence_threshold

class DetectionResults:
    """Corresponds to the DetectionResults class in the diagram."""
    def __init__(self, detections: List[str], processing_time: float):
        self.detections = detections # Simplified list of detections
        self.processing_time = processing_time #

class Analysis:
    """Corresponds to the Analysis class in the diagram."""
    def __init__(self, filename: str, processed_time: datetime.datetime, results: DetectionResults):
        self.analysis_id = f"id_{random.randint(1000, 9999)}"
        self.image_path = filename #
        self.timestamp = processed_time #
        self.results = results #

# --- State Management (from Class Diagram) ---

class Session:
    """Corresponds to the Session class in the diagram."""
    def __init__(self):
        self.session_id = f"sess_{random.randint(1000, 9999)}" #
        self.start_time = datetime.datetime.now() #
        self.user_settings = UserSettings(confidence_threshold=70.0) #
        self.analyses: List[Analysis] = [] #
        self._create_mock_history()

    def _create_mock_history(self):
        # ... (implementation remains the same) ...
        history_data = [
            {"filename": "traffic_intersection_001.jpg", "processed": "2024-01-15 14:30:22", "objects": 8, "confidence": 94},
            {"filename": "highway_scene_042.jpg", "processed": "2024-01-15 14:28:15", "objects": 12, "confidence": 87},
            {"filename": "parking_lot_analysis.jpg", "processed": "2024-01-15 14:25:03", "objects": 6, "confidence": 91},
            {"filename": "batch_processing_folder", "processed": "2024-01-15 14:20:45", "objects": 156, "confidence": 89},
        ]
        for item in history_data:
            mock_results = DetectionResults(
                detections=[f"object_{i}" for i in range(item["objects"])],
                processing_time=random.uniform(1.5, 4.5)
            )
            self.analyses.append(
                Analysis(
                    filename=item["filename"],
                    processed_time=datetime.datetime.strptime(item["processed"], "%Y-%m-%d %H:%M:%S"),
                    results=mock_results
                )
            )

    def get_history(self) -> List[Analysis]:
        return sorted(self.analyses, key=lambda x: x.timestamp, reverse=True)

    def add_analysis(self, analysis: Analysis):
        self.analyses.append(analysis)
        
    def get_user_settings(self) -> UserSettings:
        return self.user_settings

# --- Mock Backend & Managers (from Class Diagram) ---
class MockAPIClient:
    """Mock APIClient class."""
    def __init__(self, session: Session):
        self.session = session

    def upload_image(self, page: ft.Page, file: Optional[ft.FilePickerResultEvent]):
        """Simulates uploading an image and getting a result."""
        if not file or not file.files:
            page.snack_bar = ft.SnackBar(ft.Text("No file selected!"), bgcolor=ft.Colors.RED_500)
            page.snack_bar.open = True
            page.update()
            return

        filename = file.files[0].name
        page.snack_bar = ft.SnackBar(ft.Text(f"Analyzing {filename}..."))
        page.snack_bar.open = True
        page.update()
        
        # import time; time.sleep(1) # Simulate delay

        mock_results = DetectionResults(detections=[f"object_{i}" for i in range(random.randint(3, 15))], processing_time=random.uniform(1.5, 4.5))
        new_analysis = Analysis(filename=filename, processed_time=datetime.datetime.now(), results=mock_results)
        self.session.add_analysis(new_analysis)
        
        page.snack_bar = ft.SnackBar(ft.Text(f"Analysis complete for {filename}!"), bgcolor=ft.Colors.GREEN_700)
        page.snack_bar.open = True
        page.update()
        # Optionally navigate back or show results differently for a full view
        # page.go("/") 

    def batch_process(self, page: ft.Page, files: Optional[ft.FilePickerResultEvent]):
        """Simulates batch processing images."""
        if not files or not files.files:
            page.snack_bar = ft.SnackBar(ft.Text("No files selected for batch!"), bgcolor=ft.Colors.RED_500)
            page.snack_bar.open = True
            page.update()
            return
            
        num_files = len(files.files)
        page.snack_bar = ft.SnackBar(ft.Text(f"Analyzing batch of {num_files} images..."))
        page.snack_bar.open = True
        page.update()

        # import time; time.sleep(1) # Simulate delay
        
        total_detections = []
        total_time = 0
        for i in range(num_files):
            detections_count = random.randint(3, 15)
            total_detections.extend([f"object_{i}" for i in range(detections_count)])
            total_time += random.uniform(1.5, 4.5)
        mock_results = DetectionResults(detections=total_detections, processing_time=total_time)
        new_analysis = Analysis(filename=f"batch_process_{num_files}_files.jpg", processed_time=datetime.datetime.now(), results=mock_results)
        self.session.add_analysis(new_analysis)

        page.snack_bar = ft.SnackBar(ft.Text(f"Batch analysis complete!"), bgcolor=ft.Colors.GREEN_700)
        page.snack_bar.open = True
        page.update()
        # Optionally navigate back or show results differently for a full view
        # page.go("/")

class ExportManager:
    """Corresponds to the ExportManager class."""
    def export_pdf(self, page: ft.Page, analysis: Analysis):
        """Simulates exporting analysis to PDF."""
        page.snack_bar = ft.SnackBar(ft.Text(f"Exporting {analysis.image_path} to PDF... (simulation)"))
        page.snack_bar.open = True
        page.update()
        
    def export_csv(self, page: ft.Page, analysis: Analysis):
        """Simulates exporting analysis to CSV."""
        page.snack_bar = ft.SnackBar(ft.Text(f"Exporting {analysis.image_path} to CSV... (simulation)"))
        page.snack_bar.open = True
        page.update()

# --- UI View Logic Class ---
class AppLogic:
    """Holds application state and view building logic."""
    
    def __init__(self, session: Session): 
        self.session = session 
        self.api_client = MockAPIClient(session=session) 
        self.export_manager = ExportManager() 
    
        self.single_upload_result_text = ft.Text("", color=ft.Colors.GREY_600, size=12, italic=True)
        self.batch_upload_result_text = ft.Text("", color=ft.Colors.GREY_600, size=12, italic=True)
        self.confidence_slider_value_text = ft.Text(
            f"Confidence Threshold: {int(self.session.get_user_settings().confidence_threshold)}%", 
            weight=ft.FontWeight.BOLD
        )

    # --- View Builders ---
    def build_main_menu_view(self, page: ft.Page, open_settings_dialog_func):
        """Builds the main dashboard/menu view."""
        
        def menu_card(icon, title, desc, color, on_click=None):
             # ... (menu_card definition remains the same) ...
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
                 # FIX: Changed on_click to navigate to the new route
                menu_card(ft.Icons.CAMERA_ALT, "Analyze Image", "Upload a single traffic image...", ft.Colors.BLUE_600, 
                          on_click=lambda e: page.go("/analyze_single_image")), 
                menu_card(ft.Icons.COLLECTIONS, "Analyze Multiple", "Process a batch of images...", ft.Colors.GREEN_600, 
                          on_click=lambda e: page.go("/analyze_multiple_images")), 
                menu_card(ft.Icons.HISTORY, "Check History", "View previous results...", ft.Colors.PURPLE_600, 
                          on_click=lambda e: page.go("/check_history")), 
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=40,
        )

        return ft.View("/", [
                ft.Row([
                        ft.Container(), # Spacer
                        ft.Column([
                                ft.Row([
                                        ft.Container(content=ft.Icon(ft.Icons.PSYCHOLOGY, size=40, color=ft.Colors.BLACK), margin=ft.margin.only(right=10)),
                                        ft.Column([
                                                ft.Text("TrafficVision", size=26, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK),
                                                ft.Text("AI-Powered Traffic Image Analysis", size=14, color=ft.Colors.GREY_600),
                                            ], spacing=0, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.START),
                                    ], alignment=ft.MainAxisAlignment.CENTER),
                            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, tooltip="Settings", 
                                      on_click=open_settings_dialog_func), 
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(height=60, color="transparent"),
                menu_cards,
            ], vertical_alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    # FIX: NEW VIEW BUILDER for Single Image Analysis
    def build_analyze_single_image_view(self, page: ft.Page, 
                                        single_file_picker: ft.FilePicker, 
                                        open_settings_dialog_func):
        """Builds the 'Analyze Single Image' view."""
        
        def on_analyze_click(e):
            # Pass page to api client method
            self.api_client.upload_image(page, getattr(single_file_picker, 'result', None))
            # Consider if you want to navigate back automatically after analysis
            # page.go("/")

        analyze_button = ft.ElevatedButton(
            "Analyze Image", width=300, bgcolor=ft.Colors.BLUE_600, color=ft.Colors.WHITE,
            on_click=on_analyze_click
        )

        upload_box = ft.Container(
            content=ft.Column([
                    ft.Icon(ft.Icons.UPLOAD_FILE, size=40, color=ft.Colors.GREY_600),
                    ft.Text("Upload an image", weight=ft.FontWeight.BOLD),
                    ft.Text("PNG, JPG or JPEG", color=ft.Colors.GREY_500, size=12),
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
            border=ft.border.all(1, ft.Colors.GREY_300), border_radius=12, padding=40, width=400, height=180,
            on_click=lambda e: single_file_picker.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE) 
        )

        card = ft.Container(
             content=ft.Column([
                    ft.Row([
                            ft.Icon(ft.Icons.CAMERA_ALT, color=ft.Colors.BLUE_600), 
                            ft.Text("Analyze Single Image", size=18, weight=ft.FontWeight.BOLD)
                        ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                    ft.Text("Upload a traffic image for AI-powered object detection...",
                            text_align=ft.TextAlign.CENTER, color=ft.Colors.GREY_600, size=13),
                    ft.Divider(height=20, color="transparent"), 
                    upload_box, 
                    self.single_upload_result_text, # Display selected file name here
                    ft.Divider(height=10, color="transparent"), 
                    analyze_button,
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
            width=500, padding=30, border_radius=20, bgcolor=ft.Colors.WHITE,
            shadow=ft.BoxShadow(blur_radius=12, color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK)),
            alignment=ft.alignment.center,
        )

        return ft.View(
            "/analyze_single_image", # The route for this view
            [
                ft.Row(
                    [
                        ft.TextButton("← Back to Menu", on_click=lambda e: page.go("/")), 
                        ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, on_click=open_settings_dialog_func), 
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Column( # Center the card vertically and horizontally
                    [
                        ft.Container(content=card, alignment=ft.alignment.center, expand=True)
                    ], 
                    alignment=ft.MainAxisAlignment.CENTER, 
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER, 
                    expand=True,
                ),
            ]
        )


    def build_check_history_view(self, page: ft.Page, open_settings_dialog_func):
        # ... (implementation remains the same) ...
        def history_card(analysis_item: Analysis):
            avg_confidence = random.randint(85, 99)
            return ft.Container(
                content=ft.Row([
                        ft.Column([
                                ft.Text(analysis_item.image_path, weight=ft.FontWeight.BOLD),
                                ft.Text(f"Processed: {analysis_item.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", color=ft.Colors.GREY_600, size=12),
                                ft.Text(f"Objects detected: {len(analysis_item.results.detections)} | Avg confidence: {avg_confidence}%", color=ft.Colors.GREY_600, size=12),
                            ], expand=True, spacing=2),
                        ft.Container(content=ft.Text("completed", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_GREY_700), bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_GREY_400), border_radius=20, padding=ft.padding.symmetric(5, 10), alignment=ft.alignment.center),
                        ft.Row([
                                ft.IconButton(ft.Icons.VISIBILITY, tooltip="View"),
                                ft.IconButton(ft.Icons.DOWNLOAD, tooltip="Export", on_click=lambda e, item=analysis_item: self.export_manager.export_pdf(page, item)), 
                            ], alignment=ft.MainAxisAlignment.END, spacing=4),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=ft.Colors.WHITE, border_radius=12, padding=15, margin=ft.margin.symmetric(vertical=5),
                shadow=ft.BoxShadow(blur_radius=8, color=ft.Colors.with_opacity(0.15, ft.Colors.BLACK)),
            )

        title_section = ft.Column([
                ft.Row([ft.Icon(ft.Icons.HISTORY, color=ft.Colors.PURPLE_600), ft.Text("Analysis History", size=18, weight=ft.FontWeight.BOLD)], spacing=10, alignment=ft.MainAxisAlignment.START),
                ft.Text("View and manage your previous traffic image analyses", color=ft.Colors.GREY_600, size=13),
            ], spacing=4)

        history_list = self.session.get_history()
        
        return ft.View("/check_history", [
                ft.Row([
                        ft.TextButton("← Back to Menu", on_click=lambda e: page.go("/")), 
                        ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, on_click=open_settings_dialog_func), 
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(
                    content=ft.Column([title_section] + [history_card(item) for item in history_list], spacing=10, expand=True, scroll=ft.ScrollMode.ADAPTIVE),
                    width=700, padding=20, bgcolor=ft.Colors.WHITE, border_radius=16,
                    shadow=ft.BoxShadow(blur_radius=12, color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK)), alignment=ft.alignment.top_center,
                ),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, vertical_alignment=ft.MainAxisAlignment.START)


    def build_analyze_multiple_images_view(self, page: ft.Page, 
                                           batch_file_picker: ft.FilePicker, 
                                           open_settings_dialog_func):
        # ... (implementation remains the same) ...
        analyze_button = ft.ElevatedButton(
            "Analyze Batch", width=300, bgcolor=ft.Colors.GREEN_600, color=ft.Colors.WHITE,
            on_click=lambda e: self.api_client.batch_process(page, getattr(batch_file_picker, 'result', None)) 
        )

        upload_box = ft.Container(
            content=ft.Column([
                    ft.Icon(ft.Icons.COLLECTIONS, size=40, color=ft.Colors.GREY_600),
                    ft.Text("Upload multiple images", weight=ft.FontWeight.BOLD),
                    ft.Text("PNG, JPG or JPEG files", color=ft.Colors.GREY_500, size=12),
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
            border=ft.border.all(1, ft.Colors.GREY_300), border_radius=12, padding=40, width=400, height=180,
            on_click=lambda e: batch_file_picker.pick_files(allow_multiple=True, file_type=ft.FilePickerFileType.IMAGE) 
        )

        card = ft.Container(
            content=ft.Column([
                    ft.Row([ft.Icon(ft.Icons.COLLECTIONS, color=ft.Colors.GREEN_600), ft.Text("Analyze Multiple Images", size=18, weight=ft.FontWeight.BOLD)], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                    ft.Text("Upload a batch of traffic images...", text_align=ft.TextAlign.CENTER, color=ft.Colors.GREY_600, size=13),
                    ft.Divider(height=20, color="transparent"),
                    upload_box,
                    self.batch_upload_result_text, 
                    ft.Divider(height=10, color="transparent"),
                    analyze_button,
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
            width=500, padding=30, border_radius=20, bgcolor=ft.Colors.WHITE,
            shadow=ft.BoxShadow(blur_radius=12, color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK)), alignment=ft.alignment.center,
        )

        return ft.View("/analyze_multiple_images", [
                ft.Row([
                        ft.TextButton("← Back to Menu", on_click=lambda e: page.go("/")), 
                        ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, on_click=open_settings_dialog_func), 
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Column([ft.Container(content=card, alignment=ft.alignment.center, expand=True)], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
            ])


    # --- Dialog Content Builders & File Picker Handlers ---
    # FIX: Renamed methods, they only build content now
    def build_settings_dialog_content(self, page: ft.Page, close_dialog_func):
        """Builds the *content* for the settings dialog."""
        
        def on_slider_changed(e):
            # ... (implementation remains the same) ...
            if not page: return 
            new_value = int(e.control.value)
            self.session.get_user_settings().confidence_threshold = new_value
            self.confidence_slider_value_text.value = f"Confidence Threshold: {new_value}%"
            if page.dialog and hasattr(page.dialog, 'content') and isinstance(page.dialog.content, ft.Control) and callable(page.dialog.content.update):
                 page.dialog.content.update()
            else:
                 page.update() # Fallback

        initial_value = self.session.get_user_settings().confidence_threshold
        
        return ft.Column(
                [
                    ft.Text("Configure your AI model parameters...", color=ft.Colors.GREY_700, size=13),
                    ft.Divider(),
                    self.confidence_slider_value_text, 
                    ft.Slider(
                        min=0, max=100, value=initial_value, divisions=100, 
                        active_color=ft.Colors.BLACK, inactive_color=ft.Colors.GREY_300, 
                        on_change=on_slider_changed, 
                    ),
                    ft.Text("Minimum confidence level required...", color=ft.Colors.GREY_600, size=12),
                ], tight=True, spacing=10, width=400
            )

    # FIX: Removed _build_analyze_single_dialog_content as it's not needed with a full view

    # --- File Picker Handlers ---
    def on_single_file_result(self, page: ft.Page, e: ft.FilePickerResultEvent):
        """Handler for when a single file is picked."""
        if e.files: self.single_upload_result_text.value = f"Selected file: {e.files[0].name}"
        else: self.single_upload_result_text.value = "No file selected."
        # FIX: Update the whole page since it's a full view now
        page.update() 

    def on_batch_file_result(self, page: ft.Page, e: ft.FilePickerResultEvent):
        """Handler for when multiple files are picked."""
        if e.files: self.batch_upload_result_text.value = f"Selected {len(e.files)} files"
        else: self.batch_upload_result_text.value = "No files selected."
        page.update() 


# --- Main App Entry Point ---

def main(page: ft.Page):
    page.title = "TrafficVision - AI-Powered Traffic Image Analysis"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30
    page.scroll = "adaptive" # May need adjustment depending on view content

    # 1. Create the Session and AppLogic
    app_session = Session()
    app_logic = AppLogic(session=app_session)

    # 2. Create Dialogs and File Pickers (managed by 'main' scope)
    
    # -- Settings Dialog --
    settings_dialog = ft.AlertDialog(
        modal=True, 
        title=ft.Text("Settings", size=18, weight=ft.FontWeight.BOLD),
        actions_alignment=ft.MainAxisAlignment.END
    )
    def close_settings_dialog(e=None):
        settings_dialog.open = False
        page.update()
    settings_dialog.actions = [ft.TextButton("Close", on_click=close_settings_dialog)]
    # Build content using AppLogic method, passing necessary functions
    settings_dialog.content = app_logic.build_settings_dialog_content(page, close_settings_dialog)
    
    def open_settings_dialog(e=None):
        page.dialog = settings_dialog
        # Re-build content in case state changed (like slider value)
        settings_dialog.content = app_logic.build_settings_dialog_content(page, close_settings_dialog) 
        settings_dialog.open = True
        page.update()

    # -- File Pickers --
    single_file_picker = ft.FilePicker(on_result=lambda e: app_logic.on_single_file_result(page, e))
    batch_file_picker = ft.FilePicker(on_result=lambda e: app_logic.on_batch_file_result(page, e))

    # Add overlay controls directly to the page
    page.overlay.extend([
        single_file_picker, 
        batch_file_picker,
        # Settings Dialog is added when opened via page.dialog
    ])

    # 3. Define Route Change Handler (using AppLogic to build views)
    def route_change_handler(route): # Flet passes route string
        current_route = page.route # Get actual route from page
        page.views.clear()
        
        # Call AppLogic build methods, passing necessary page and functions/objects
        if current_route == "/":
            view = app_logic.build_main_menu_view(page, open_settings_dialog)
        # FIX: Added route for single image analysis
        elif current_route == "/analyze_single_image":
             view = app_logic.build_analyze_single_image_view(page, single_file_picker, open_settings_dialog)
        elif current_route == "/analyze_multiple_images":
             view = app_logic.build_analyze_multiple_images_view(page, batch_file_picker, open_settings_dialog)
        elif current_route == "/check_history":
             view = app_logic.build_check_history_view(page, open_settings_dialog)
        else:
            print(f"Unknown route: {current_route}, redirecting to home.")
            view = app_logic.build_main_menu_view(page, open_settings_dialog)
            if page.route != "/":
                 page.go("/") 
                 return 
                 
        page.views.append(view)
        page.update()

    # 4. Define View Pop Handler
    def view_pop_handler(e):
        page.views.pop()
        top_view_route = page.views[-1].route if page.views else "/"
        page.go(top_view_route)

    # 5. Assign Handlers and Navigate
    page.on_route_change = route_change_handler 
    page.on_view_pop = view_pop_handler
    page.go(page.route)


if __name__ == "__main__":
    ft.app(target=main)