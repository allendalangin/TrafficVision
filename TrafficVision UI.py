import flet as ft

# --- Check History View ---
def check_history_view(page: ft.Page, open_settings):
    # Sample data (can be replaced with actual database or logs later)
    history_data = [
        {
            "filename": "traffic_intersection_001.jpg",
            "status": "completed",
            "processed": "2024-01-15 14:30:22",
            "objects": 8,
            "confidence": 94,
        },
        {
            "filename": "highway_scene_042.jpg",
            "status": "completed",
            "processed": "2024-01-15 14:28:15",
            "objects": 12,
            "confidence": 87,
        },
        {
            "filename": "parking_lot_analysis.jpg",
            "status": "completed",
            "processed": "2024-01-15 14:25:03",
            "objects": 6,
            "confidence": 91,
        },
        {
            "filename": "batch_processing_folder",
            "status": "completed",
            "processed": "2024-01-15 14:20:45",
            "objects": 156,
            "confidence": 89,
        },
    ]

    # Function to generate a card for each history entry
    def history_card(item):
        return ft.Container(
            content=ft.Row(
                [
                    ft.Column(
                        [
                            ft.Text(item["filename"], weight=ft.FontWeight.BOLD),
                            ft.Text(
                                f"Processed: {item['processed']}",
                                color=ft.Colors.GREY_600,
                                size=12,
                            ),
                            ft.Text(
                                f"Objects detected: {item['objects']} | Avg confidence: {item['confidence']}%",
                                color=ft.Colors.GREY_600,
                                size=12,
                            ),
                        ],
                        expand=True,
                        spacing=2,
                    ),
                    ft.Container(
                        content=ft.Text(
                            item["status"],
                            size=12,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_GREY_700,
                        ),
                        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLUE_GREY_400),
                        border_radius=20,
                        padding=ft.padding.symmetric(5, 10),
                        alignment=ft.alignment.center,
                    ),
                    ft.Row(
                        [
                            ft.IconButton(ft.Icons.VISIBILITY, tooltip="View"),
                            ft.IconButton(ft.Icons.DOWNLOAD, tooltip="Export"),
                        ],
                        alignment=ft.MainAxisAlignment.END,
                        spacing=4,
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=ft.Colors.WHITE,
            border_radius=12,
            padding=15,
            margin=ft.margin.symmetric(vertical=5),
            shadow=ft.BoxShadow(
                blur_radius=8,
                color=ft.Colors.with_opacity(0.15, ft.Colors.BLACK),
            ),
        )

    # --- Title + Description ---
    title_section = ft.Column(
        [
            ft.Row(
                [
                    ft.Icon(ft.Icons.HISTORY, color=ft.Colors.PURPLE_600),
                    ft.Text("Analysis History", size=18, weight=ft.FontWeight.BOLD),
                ],
                spacing=10,
                alignment=ft.MainAxisAlignment.START,
            ),
            ft.Text(
                "View and manage your previous traffic image analyses",
                color=ft.Colors.GREY_600,
                size=13,
            ),
        ],
        spacing=4,
    )

    # --- Page Layout ---
    return ft.View(
        "/check_history",
        [
            ft.Row(
                [
                    ft.TextButton("← Back to Menu", on_click=lambda e: page.go("/")),
                    ft.IconButton(
                        ft.Icons.SETTINGS,
                        icon_color=ft.Colors.GREY_700,
                        on_click=open_settings,
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            ft.Container(
                content=ft.Column(
                    [title_section] + [history_card(item) for item in history_data],
                    spacing=10,
                    expand=True,
                ),
                width=700,
                padding=20,
                bgcolor=ft.Colors.WHITE,
                border_radius=16,
                shadow=ft.BoxShadow(
                    blur_radius=12,
                    color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK),
                ),
                alignment=ft.alignment.top_center,
            ),
        ],
        scroll=ft.ScrollMode.AUTO,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        vertical_alignment=ft.MainAxisAlignment.START,
    )

# --- Analyze Single Image View ---
def analyze_single_image_view(page: ft.Page, open_settings):
    upload = ft.FilePicker()
    upload_result = ft.Text("", color=ft.Colors.GREY_600, size=12, italic=True)

    def on_upload_result(e: ft.FilePickerResultEvent):
        if e.files:
            upload_result.value = f"Selected file: {e.files[0].name}"
        else:
            upload_result.value = "No file selected."
        page.update()

    upload.on_result = on_upload_result
    page.overlay.append(upload)

    analyze_button = ft.ElevatedButton(
        "Analyze Image",
        width=300,
        bgcolor=ft.Colors.BLUE_600,
        color=ft.Colors.WHITE,
        on_click=lambda e: page.snack_bar.open(ft.SnackBar(ft.Text("Analyzing image..."))),
    )

    upload_box = ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.Icons.UPLOAD_FILE, size=40, color=ft.Colors.GREY_600),
                ft.Text("Upload an image", weight=ft.FontWeight.BOLD),
                ft.Text("PNG, JPG or JPEG", color=ft.Colors.GREY_500, size=12),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5,
        ),
        border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=12,
        padding=40,
        width=400,
        height=180,
        on_click=lambda e: upload.pick_files(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.IMAGE,
        ),
    )

    card = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.CAMERA_ALT, color=ft.Colors.BLUE_600),
                        ft.Text("Analyze Single Image", size=18, weight=ft.FontWeight.BOLD),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10,
                ),
                ft.Text(
                    "Upload a traffic image for AI-powered object detection and classification",
                    text_align=ft.TextAlign.CENTER,
                    color=ft.Colors.GREY_600,
                    size=13,
                ),
                ft.Divider(height=20, color="transparent"),
                upload_box,
                upload_result,
                ft.Divider(height=10, color="transparent"),
                analyze_button,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        ),
        width=500,
        padding=30,
        border_radius=20,
        bgcolor=ft.Colors.WHITE,
        shadow=ft.BoxShadow(
            blur_radius=12,
            color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK),
        ),
        alignment=ft.alignment.center,
    )

    return ft.View(
        "/analyze_single_image",
        [
            ft.Row(
                [
                    ft.TextButton("← Back to Menu", on_click=lambda e: page.go("/")),
                    ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, on_click=open_settings),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            ft.Column(
                [
                    ft.Container(content=card, alignment=ft.alignment.center, expand=True)
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True,
            ),
        ],
    )


# --- Analyze Multiple Images View ---
def analyze_multiple_images_view(page: ft.Page, open_settings):
    upload = ft.FilePicker()
    upload_result = ft.Text("", color=ft.Colors.GREY_600, size=12, italic=True)

    def on_upload_result(e: ft.FilePickerResultEvent):
        if e.files:
            upload_result.value = f"Selected {len(e.files)} files"
        else:
            upload_result.value = "No files selected."
        page.update()

    upload.on_result = on_upload_result
    page.overlay.append(upload)

    analyze_button = ft.ElevatedButton(
        "Analyze Batch",
        width=300,
        bgcolor=ft.Colors.GREEN_600,
        color=ft.Colors.WHITE,
        on_click=lambda e: page.snack_bar.open(ft.SnackBar(ft.Text("Analyzing multiple images..."))),
    )

    upload_box = ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.Icons.COLLECTIONS, size=40, color=ft.Colors.GREY_600),
                ft.Text("Upload multiple images", weight=ft.FontWeight.BOLD),
                ft.Text("PNG, JPG or JPEG files", color=ft.Colors.GREY_500, size=12),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5,
        ),
        border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=12,
        padding=40,
        width=400,
        height=180,
        on_click=lambda e: upload.pick_files(
            allow_multiple=True,
            file_type=ft.FilePickerFileType.IMAGE,
        ),
    )

    card = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.COLLECTIONS, color=ft.Colors.GREEN_600),
                        ft.Text("Analyze Multiple Images", size=18, weight=ft.FontWeight.BOLD),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10,
                ),
                ft.Text(
                    "Upload a batch of traffic images for bulk processing and analysis",
                    text_align=ft.TextAlign.CENTER,
                    color=ft.Colors.GREY_600,
                    size=13,
                ),
                ft.Divider(height=20, color="transparent"),
                upload_box,
                upload_result,
                ft.Divider(height=10, color="transparent"),
                analyze_button,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        ),
        width=500,
        padding=30,
        border_radius=20,
        bgcolor=ft.Colors.WHITE,
        shadow=ft.BoxShadow(blur_radius=12, color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK)),
        alignment=ft.alignment.center,
    )

    return ft.View(
        "/analyze_multiple_images",
        [
            ft.Row(
                [
                    ft.TextButton("← Back to Menu", on_click=lambda e: page.go("/")),
                    ft.IconButton(ft.Icons.SETTINGS, icon_color=ft.Colors.GREY_700, on_click=open_settings),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            ft.Column(
                [
                    ft.Container(content=card, alignment=ft.alignment.center, expand=True)
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True,
            ),
        ],
    )


# --- Main App ---
def main(page: ft.Page):
    page.title = "TrafficVision - AI-Powered Traffic Image Analysis"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30
    page.scroll = "adaptive"

    # --- Settings Dialog Setup ---
    confidence_value = ft.Text("Confidence Threshold: 70%", weight=ft.FontWeight.BOLD)

    def slider_changed(e):
        confidence_value.value = f"Confidence Threshold: {int(e.control.value)}%"
        page.update()

    # ✅ Define open/close before the dialog so they're available
    def close_settings(e=None):
        settings_dialog.open = False
        page.update()

    def open_settings(e=None):
        page.dialog = settings_dialog
        settings_dialog.open = True
        page.update()

    # --- Define the dialog AFTER the functions ---
    settings_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Settings", size=18, weight=ft.FontWeight.BOLD),
        content=ft.Column(
            [
                ft.Text("Configure your AI model parameters for traffic analysis.",
                        color=ft.Colors.GREY_700, size=13),
                ft.Divider(),
                confidence_value,
                ft.Slider(
                    min=0,
                    max=100,
                    value=70,
                    divisions=100,
                    active_color=ft.Colors.BLACK,
                    inactive_color=ft.Colors.GREY_300,
                    on_change=slider_changed,
                ),
                ft.Text(
                    "Minimum confidence level required for object detection",
                    color=ft.Colors.GREY_600,
                    size=12,
                ),
            ],
            tight=True,
            spacing=10,
            width=400,
        ),
        actions_alignment=ft.MainAxisAlignment.END,
        actions=[ft.TextButton("Close", on_click=close_settings)],
    )

    page.overlay.append(settings_dialog)

    # --- Header ---
    def get_header():
        return ft.Container(
            content=ft.Row(
                [
                    ft.Container(
                        content=ft.Icon(ft.Icons.PSYCHOLOGY, size=40, color=ft.Colors.BLUE_GREY_900),
                        bgcolor=ft.Colors.BLUE_GREY_50,
                        padding=10,
                        border_radius=12,
                    ),
                    ft.Column(
                        [
                            ft.Text("TrafficVision", size=26, weight=ft.FontWeight.BOLD),
                            ft.Text("AI-Powered Traffic Image Analysis", size=13, color=ft.Colors.GREY_600),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=2,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            ),
            alignment=ft.alignment.center,
            padding=20,
        )



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

    # --- Navigation functions ---
    def go_to_single(e): page.go("/analyze_single_image")
    def go_to_multiple(e): page.go("/analyze_multiple_images")
    def go_to_history(e): page.go("/check_history")

    # --- Menu Cards must be defined BEFORE main_menu ---
    menu_cards = ft.Row(
        [
            menu_card(ft.Icons.CAMERA_ALT, "Analyze Image", "Upload a single traffic image for AI-powered object object detection and classification", ft.Colors.BLUE_600, on_click=go_to_single),
            menu_card(ft.Icons.COLLECTIONS, "Analyze Multiple Images", "Process a batch of traffic images for bulk analysis and reporting", ft.Colors.GREEN_600, on_click=go_to_multiple),
            menu_card(ft.Icons.HISTORY, "Check History", "View and manage your previous analysis results and reports", ft.Colors.PURPLE_600, on_click=go_to_history),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=40,
    )

    # --- Now safe to create main_menu ---
    main_menu = ft.View(
        "/",
        [
            # Top row with centered header and right-aligned settings button
            ft.Row(
                [
                    ft.Container(),  # Empty container for spacing on the left
                    ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Container(
                                        content=ft.Icon(
                                            ft.Icons.PSYCHOLOGY,
                                            size=40,
                                            color=ft.Colors.BLACK,
                                        ),
                                        margin=ft.margin.only(right=10),
                                    ),  
                                    ft.Column(
                                        [
                                            ft.Text(
                                                "TrafficVision",
                                                size=26,
                                                weight=ft.FontWeight.BOLD,
                                                color=ft.Colors.BLACK,
                                            ),
                                            ft.Text(
                                                "AI-Powered Traffic Image Analysis",
                                                size=14,
                                                color=ft.Colors.GREY_600,
                                            ),
                                        ],
                                        spacing=0,
                                        alignment=ft.MainAxisAlignment.CENTER,
                                        horizontal_alignment=ft.CrossAxisAlignment.START,
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.IconButton(
                        ft.Icons.SETTINGS,
                        icon_color=ft.Colors.GREY_700,
                        tooltip="Settings",
                        on_click=open_settings,
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            ft.Divider(height=60, color="transparent"),
            menu_cards,
        ],
        vertical_alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )


    def route_change(e):
        page.views.clear()
        if page.route == "/":
            page.views.append(main_menu)
        elif page.route == "/analyze_single_image":
            page.views.append(analyze_single_image_view(page, open_settings))
        elif page.route == "/analyze_multiple_images":
            page.views.append(analyze_multiple_images_view(page, open_settings))
        elif page.route == "/check_history":
            page.views.append(check_history_view(page, open_settings))
        page.update()

    page.on_route_change = route_change
    page.go(page.route)

    def view_pop(e):
        page.views.pop()
        page.go(page.views[-1].route)

    page.on_view_pop = view_pop

ft.app(target=main)
