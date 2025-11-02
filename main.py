# /main.py

import flet as ft
import os
import shutil
import sqlite3
import traceback # Kept for better error handling

# Import Client Logic and UI from the new folders
from logic.core import Session, ExportManager
from ui.views import AppLogic 

# --- Directory for saving images persistently (Kept for setup) ---
PERSISTENT_IMAGE_DIR = "persistent_images"
os.makedirs(PERSISTENT_IMAGE_DIR, exist_ok=True)


async def main(page: ft.Page):
    print("\n" + "="*40)
    page.title = "TrafficVision - AI-Powered Traffic Image Analysis"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30

    # 1. Initialize Core State and Managers
    app_session = Session()

    def on_save_file_result(e: ft.FilePickerResultEvent):
        if e.path:
            try:
                # Assuming data_to_save and save_mode are set on the FilePicker instance
                data_to_save = e.control.data_to_save
                save_mode = getattr(e.control, "save_mode", "text")
                
                if save_mode == "binary":
                    with open(e.path, "wb") as f:
                        f.write(data_to_save)
                else:
                    with open(e.path, "w", encoding="utf-8") as f:
                        f.write(data_to_save)
                        
                page.snack_bar = ft.SnackBar(ft.Text(f"Successfully exported to {e.path}"), bgcolor=ft.Colors.GREEN_700)
            except Exception as ex:
                print(f"--- DEBUG: Error saving file: {ex} ---")
                page.snack_bar = ft.SnackBar(ft.Text(f"Error saving file: {ex}"), bgcolor=ft.Colors.RED_500)
        else:
            page.snack_bar = ft.SnackBar(ft.Text("Save operation cancelled."))
        
        page.snack_bar.open = True
        page.update()

    save_file_dialog = ft.FilePicker(on_result=on_save_file_result)
    
    export_manager = ExportManager(save_dialog=save_file_dialog)
    app_logic = AppLogic(session=app_session, export_manager=export_manager)

    # 2. Create File Pickers and Overlay
    single_file_picker = ft.FilePicker(on_result=app_logic.on_single_file_result)
    batch_file_picker = ft.FilePicker(on_result=app_logic.on_batch_file_result)

    page.overlay.extend([
        single_file_picker, 
        batch_file_picker,
        save_file_dialog 
    ])

    # 3. Define Route Change Handler
    async def route_change_handler(e: ft.RouteChangeEvent):
        current_page = e.page
        current_route = e.route
        current_page.views.clear()
        
        try:
            # Logic for route to view mapping (now calls methods on app_logic)
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
                    current_page.go("/check_history")
                    return
            
            elif current_route.startswith("/history/batch/"):
                batch_id = current_route.split("/")[-1]
                batch = app_session.db.load_batch_by_id(batch_id)
                if batch:
                    view = app_logic.build_batch_history_detail_view(current_page, batch)
                else:
                    current_page.go("/check_history")
                    return
            
            else:
                view = app_logic.build_main_menu_view(current_page)
                if current_page.route != "/":
                    current_page.go("/")
                    return
            
            current_page.views.append(view)
            current_page.update()
            
        except Exception:
            print("!!! --- CRITICAL ERROR in route_change_handler: --- !!!")
            traceback.print_exc()

    # 4. Define View Pop Handler
    def view_pop_handler(e: ft.ViewPopEvent):
        e.page.views.pop()
        top_view_route = e.page.views[-1].route if e.page.views else "/"
        e.page.go(top_view_route)

    # 5. Assign Handlers and Navigate
    page.on_route_change = route_change_handler
    page.on_view_pop = view_pop_handler
    page.go(page.route) 


if __name__ == "__main__":
    ft.app(target=main)