import flet as ft

# --- Analyze Single Image View ---
def analyze_single_image_view(page: ft.Page, open_settings):
    upload = ft.FilePicker()
    upload_result = ft.Text("", color=ft.colors.GREY_600, size=12, italic=True)

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
        bgcolor=ft.colors.BLUE_600,
        color=ft.colors.WHITE,
        on_click=lambda e: page.snack_bar.open(ft.SnackBar(ft.Text("Analyzing image..."))),
    )

    upload_box = ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.icons.UPLOAD_FILE, size=40, color=ft.colors.GREY_600),
                ft.Text("Upload an image", weight=ft.FontWeight.BOLD),
                ft.Text("PNG, JPG or JPEG", color=ft.colors.GREY_500, size=12),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5,
        ),
        border=ft.border.all(1, ft.colors.GREY_300),
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
                        ft.Icon(ft.icons.CAMERA_ALT, color=ft.colors.BLUE_600),
                        ft.Text("Analyze Single Image", size=18, weight=ft.FontWeight.BOLD),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10,
                ),
                ft.Text(
                    "Upload a traffic image for AI-powered object detection and classification",
                    text_align=ft.TextAlign.CENTER,
                    color=ft.colors.GREY_600,
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
        bgcolor=ft.colors.WHITE,
        shadow=ft.BoxShadow(
            blur_radius=12,
            color=ft.colors.with_opacity(0.2, ft.colors.BLACK),
        ),
        alignment=ft.alignment.center,
    )

    return ft.View(
        "/analyze_single_image",
        [
            ft.Row(
                [
                    ft.TextButton("← Back to Menu", on_click=lambda e: page.go("/")),
                    ft.IconButton(ft.icons.SETTINGS, icon_color=ft.colors.GREY_700, on_click=open_settings),
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
    upload_result = ft.Text("", color=ft.colors.GREY_600, size=12, italic=True)

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
        bgcolor=ft.colors.GREEN_600,
        color=ft.colors.WHITE,
        on_click=lambda e: page.snack_bar.open(ft.SnackBar(ft.Text("Analyzing multiple images..."))),
    )

    upload_box = ft.Container(
        content=ft.Column(
            [
                ft.Icon(ft.icons.COLLECTIONS, size=40, color=ft.colors.GREY_600),
                ft.Text("Upload multiple images", weight=ft.FontWeight.BOLD),
                ft.Text("PNG, JPG or JPEG files", color=ft.colors.GREY_500, size=12),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5,
        ),
        border=ft.border.all(1, ft.colors.GREY_300),
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
                        ft.Icon(ft.icons.COLLECTIONS, color=ft.colors.GREEN_600),
                        ft.Text("Analyze Multiple Images", size=18, weight=ft.FontWeight.BOLD),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10,
                ),
                ft.Text(
                    "Upload a batch of traffic images for bulk processing and analysis",
                    text_align=ft.TextAlign.CENTER,
                    color=ft.colors.GREY_600,
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
        bgcolor=ft.colors.WHITE,
        shadow=ft.BoxShadow(blur_radius=12, color=ft.colors.with_opacity(0.2, ft.colors.BLACK)),
        alignment=ft.alignment.center,
    )

    return ft.View(
        "/analyze_multiple_images",
        [
            ft.Row(
                [
                    ft.TextButton("← Back to Menu", on_click=lambda e: page.go("/")),
                    ft.IconButton(ft.icons.SETTINGS, icon_color=ft.colors.GREY_700, on_click=open_settings),
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

    # Settings dialog
    confidence_slider = ft.Slider(value=0.85, min=0.5, max=1.0, divisions=10, label="{value:.2f}")
    settings_dialog = ft.AlertDialog(
        title=ft.Text("Settings"),
        content=ft.Column(
            [ft.Text("Adjust AI Confidence Threshold"), confidence_slider],
            tight=True,
        ),
        actions=[ft.TextButton("Close", on_click=lambda e: page.close(settings_dialog))],
    )

    def open_settings(e):
        page.dialog = settings_dialog
        settings_dialog.open = True
        page.update()

    def get_header():
        return ft.Row(
            [
                ft.Container(
                    content=ft.Icon(ft.icons.PSYCHOLOGY, size=40, color=ft.colors.BLUE_GREY_900),
                    bgcolor=ft.colors.BLUE_GREY_50,
                    padding=10,
                    border_radius=12,
                ),
                ft.Column(
                    [
                        ft.Text("TrafficVision", size=26, weight=ft.FontWeight.BOLD),
                        ft.Text("AI-Powered Traffic Image Analysis", color=ft.colors.GREY_600),
                    ]
                ),
                ft.IconButton(ft.icons.SETTINGS, icon_color=ft.colors.GREY_800, on_click=open_settings),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

    def menu_card(icon, title, desc, color, on_click=None):
        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Icon(icon, size=40, color=color),
                        bgcolor=ft.colors.with_opacity(0.15, color),
                        border_radius=12,
                        padding=10,
                    ),
                    ft.Text(title, size=18, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                    ft.Text(desc, size=13, text_align=ft.TextAlign.CENTER, color=ft.colors.GREY_600),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=8,
            ),
            width=250,
            height=220,
            alignment=ft.alignment.center,
            padding=20,
            bgcolor=ft.colors.WHITE,
            border_radius=16,
            shadow=ft.BoxShadow(blur_radius=8, color=ft.colors.with_opacity(0.2, ft.colors.BLACK)),
            ink=True,
            on_click=on_click,
        )

    # --- Navigation functions ---
    def go_to_single(e):
        page.go("/analyze_single_image")

    def go_to_multiple(e):
        page.go("/analyze_multiple_images")

    menu_cards = ft.Row(
        [
            menu_card(ft.icons.CAMERA_ALT, "Analyze Image",
                      "Upload a single traffic image for AI-powered object detection and classification",
                      ft.colors.BLUE_600, on_click=go_to_single),
            menu_card(ft.icons.COLLECTIONS, "Analyze Multiple Images",
                      "Process a batch of traffic images for bulk analysis and reporting",
                      ft.colors.GREEN_600, on_click=go_to_multiple),
            menu_card(ft.icons.HISTORY, "Check History",
                      "View and manage your previous analysis results and reports",
                      ft.colors.PURPLE_600),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=40,
    )

    main_menu = ft.View("/", [get_header(), ft.Divider(height=60, color="transparent"), menu_cards],
                        vertical_alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    def route_change(e):
        page.views.clear()
        if page.route == "/":
            page.views.append(main_menu)
        elif page.route == "/analyze_single_image":
            page.views.append(analyze_single_image_view(page, open_settings))
        elif page.route == "/analyze_multiple_images":
            page.views.append(analyze_multiple_images_view(page, open_settings))
        page.update()

    page.on_route_change = route_change
    page.go(page.route)

    def view_pop(e):
        page.views.pop()
        page.go(page.views[-1].route)

    page.on_view_pop = view_pop


ft.app(target=main)
