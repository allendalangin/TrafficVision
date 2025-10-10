import flet as ft

def main(page: ft.Page):
    page.title = "TrafficVision - AI-Powered Traffic Image Analysis"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30
    page.scroll = "adaptive"

    # --- Settings Dialog ---
    confidence_slider = ft.Slider(value=0.85, min=0.5, max=1.0, divisions=10, label="{value:.2f}")
    settings_dialog = ft.AlertDialog(
        title=ft.Text("Settings"),
        content=ft.Column(
            [
                ft.Text("Adjust AI Confidence Threshold"),
                confidence_slider,
            ],
            tight=True,
        ),
        actions=[ft.TextButton("Close", on_click=lambda e: page.close(settings_dialog))]
    )

    def open_settings(e):
        page.dialog = settings_dialog
        settings_dialog.open = True
        page.update()

    # --- Header ---
    header = ft.Column(
        [
            ft.Row(
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
            ),
            ft.Row(
                [
                    ft.Container(
                        content=ft.Text("System Online ✓", color=ft.colors.GREEN_700),
                        border_radius=12,
                        padding=ft.padding.all(8),
                        bgcolor=ft.colors.GREEN_50,
                    ),
                    ft.Container(
                        content=ft.Text("v2.1.0"),
                        border_radius=12,
                        padding=ft.padding.all(8),
                        bgcolor=ft.colors.GREY_100,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=10,
            ),
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10,
    )

    # --- Cards for Menu Options ---
    def menu_card(icon, title, desc, color):
        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Icon(icon, size=40, color=color),
                        bgcolor=ft.colors.with_opacity(0.15, color),
                        border_radius=12,
                        padding=10,
                    ),
                    ft.Text(title, size=18, weight=ft.FontWeight.BOLD),
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
            on_click=lambda e: page.snack_bar.open("Feature coming soon!")
        )

    # --- Menu Cards Section ---
    menu_cards = ft.Row(
        [
            menu_card(
                ft.icons.CAMERA_ALT,
                "Analyze Image",
                "Upload a single traffic image for AI-powered object detection and classification",
                ft.colors.BLUE_600,
            ),
            menu_card(
                ft.icons.COLLECTIONS,
                "Analyze Multiple Images",
                "Process a batch of traffic images for bulk analysis and reporting",
                ft.colors.GREEN_600,
            ),
            menu_card(
                ft.icons.HISTORY,
                "Check History",
                "View and manage your previous analysis results and reports",
                ft.colors.PURPLE_600,
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=40,
    )

    # --- Center Section (vertically centered layout) ---
    center_section = ft.Column(
        [
            header,
            ft.Divider(height=80, color="transparent"),
            menu_cards,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True,  # makes sure the whole thing is vertically centered
    )

    # --- Layout Assembly ---
    page.add(center_section)


ft.app(target=main)