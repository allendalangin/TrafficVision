# /ui/views.py

import flet as ft
import os
import datetime
import asyncio
from typing import List, Optional, Union

# --- Imports from Client Logic and Data Layers ---
from logic.core import Session, ExportManager, APIClient # Core functionality
from data_models import AnalysisResult, BatchAnalysis     # Data models
from ui.controls import BatchImageCard                    # Custom UI controls

# --- The AppLogic Class with all View Builders ---

class AppLogic:
    """Holds application state and view building logic."""
    
    def __init__(self, session: Session, export_manager: ExportManager): 
        self.session = session 
        self.api_client = APIClient(session=session, base_url="http://127.0.0.1:8000")
        self.export_manager = export_manager
    
        # --- Shared UI controls ---
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

    # --- Handlers (Standard sync/async functions) ---

    def on_slider_changed(self, e: ft.ControlEvent):
        new_value = int(e.control.value)
        self.session.get_user_settings().confidence_threshold = new_value
        
        text_control = e.control.parent.controls[0]
        text_control.value = f"Confidence Threshold: {new_value}%"
        
        if e.control.page:
            e.control.page.update()

    def on_settings_click(self, e: ft.ControlEvent):
        stack = e.page.views[-1].controls[0]
        settings_panel = stack.controls[1]
        
        settings_panel.visible = not settings_panel.visible
        e.page.update()

    def remove_batch_card(self, e: ft.ControlEvent):
        card_to_remove = e.control.parent.parent.parent
        
        if isinstance(card_to_remove, BatchImageCard):
            self.batch_image_grid.controls.remove(card_to_remove)
            
            count = len(self.batch_image_grid.controls)
            if count == 0:
                self.batch_upload_result_text.value = "No files selected."
            else:
                self.batch_upload_result_text.value = f"Selected {count} files"
            
            e.page.update()

    def _build_settings_panel(self) -> ft.Container:
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

    def build_analyze_single_image_view(self, page: ft.Page, single_file_picker: ft.FilePicker):
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

    # --- NEW Synchronous Wrapper for run_task ---
    def _run_delete_task(self, page: ft.Page, item: Union[AnalysisResult, BatchAnalysis]):
        """Schedules the async delete handler safely from a sync context."""
        # Use page.run_task, passing the coroutine function reference and the page/item arguments separately.
        page.run_task(self.delete_history_item, page=page, item=item)

    # --- View Builders ---
    def build_check_history_view(self, page: ft.Page):
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
                                # --- DELETE OPTION (Single) ---
                                ft.PopupMenuItem(), # Separator
                                ft.PopupMenuItem(
                                    text="Delete Analysis", 
                                    icon=ft.Icon(ft.Icons.DELETE_FOREVER, color=ft.Colors.RED_500),
                                    # FIX: Call the synchronous wrapper, passing page and item
                                    on_click=lambda e, item=analysis_item: self._run_delete_task(e.page, item) 
                                ),
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
                        ft.PopupMenuButton(
                            icon=ft.Icons.MORE_VERT,
                            items=[
                                ft.PopupMenuItem(text="Export as PDF", icon=ft.Icons.PICTURE_AS_PDF, on_click=lambda e, item=batch_item: on_export_pdf(item)),
                                ft.PopupMenuItem(text="Export as CSV", icon=ft.Icons.TABLE_ROWS, on_click=lambda e, item=batch_item: on_export_csv(item)),
                                ft.PopupMenuItem(text="Export as JSON", icon=ft.Icons.DATA_OBJECT, on_click=lambda e, item=batch_item: on_export_json(item)),
                                # --- DELETE OPTION (Batch) ---
                                ft.PopupMenuItem(), # Separator
                                ft.PopupMenuItem(
                                    text="Delete Batch", 
                                    icon=ft.Icon(ft.Icons.DELETE_FOREVER, color=ft.Colors.RED_500),
                                    # FIX: Call the synchronous wrapper, passing page and item
                                    on_click=lambda e, item=batch_item: self._run_delete_task(e.page, item)
                                ),
                            ]
                        )
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

    def build_analyze_multiple_images_view(self, page: ft.Page, batch_file_picker: ft.FilePicker):
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

    def build_single_history_detail_view(self, page: ft.Page, analysis: AnalysisResult):
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

    def build_batch_history_detail_view(self, page: ft.Page, batch: BatchAnalysis):
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
                file_path=analysis.image_name,
                file_name=os.path.basename(analysis.image_name),
                is_readonly=True
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
        current_page = e.page
        if not current_page: return
        if e.files:
            self.single_image_display.src = e.files[0].path
            self.single_image_container.content = self.single_image_display
            self.single_analysis_output_text.value = "" 
        else:
            self.single_image_container.content = self.single_image_placeholder
        current_page.update()

    def on_batch_file_result(self, e: ft.FilePickerResultEvent):
        current_page = e.page
        if not current_page: return
        
        self.batch_image_grid.controls.clear()
        self.batch_analysis_output_text.value = ""
        
        if e.files:
            self.batch_upload_result_text.value = f"Selected {len(e.files)} files"
            
            for file_obj in e.files:
                card = BatchImageCard(
                    file_path=file_obj.path, 
                    file_name=file_obj.name,
                    on_remove=self.remove_batch_card
                )
                self.batch_image_grid.controls.append(card)
        else:
            self.batch_upload_result_text.value = "No file selected."
        
        current_page.update()
    
    # --- DELETE HANDLERS ---
    async def delete_history_item(self, page: ft.Page, item: Union[AnalysisResult, BatchAnalysis]):
        # Removed 'e: ft.ControlEvent' from the signature and replaced it with 'page: ft.Page'
        
        if isinstance(item, BatchAnalysis):
            # Delete all contained analyses sequentially
            # Use a copy of IDs since the session list changes during iteration
            analysis_ids_to_delete = [analysis.analysis_id for analysis in item.analyses]
            for analysis_id in analysis_ids_to_delete:
                # We await the helper, which handles the DB and state update
                await self._execute_single_delete(page, analysis_id)
        else:
            # Single analysis deletion
            await self._execute_single_delete(page, item.analysis_id)

        # Refresh the whole history view after deletion is complete
        page.go("/check_history")

    async def _execute_single_delete(self, page: ft.Page, analysis_id: str):
        """Internal helper to execute deletion and update local state."""
        try:
            # 1. API call deletes from DB
            await self.api_client.delete_analysis_by_id(analysis_id)
            
            # 2. Session cleanup removes from in-memory list
            self.session.remove_analysis(analysis_id)
            
            page.snack_bar = ft.SnackBar(
                ft.Text(f"Analysis {analysis_id[:8]}... deleted.", color=ft.Colors.WHITE), 
                bgcolor=ft.Colors.PURPLE_600
            )
        except Exception as ex:
            page.snack_bar = ft.SnackBar(
                ft.Text(f"Deletion failed: {ex}"), 
                bgcolor=ft.Colors.RED_500
            )
        finally:
            page.snack_bar.open = True
            page.update()