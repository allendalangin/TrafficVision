# /ui/controls.py

import flet as ft

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