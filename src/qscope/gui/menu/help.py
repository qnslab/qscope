import pathlib
import webbrowser

import PyQt6
from loguru import logger
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMainWindow, QMenu, QMessageBox, QWidget


# Class for the server options
class HelpMenu(QMenu):
    def __init__(self, parent, menuBar):
        super().__init__()
        self.parent = parent
        self._create_menu(menuBar)

    def _create_menu(self, menuBar):
        # Help menu
        help_menu = menuBar.addMenu("&Help")

        help_about_action = QAction("About", self.parent)
        help_about_action.triggered.connect(self._show_about_dialog)

        help_shortcuts_action = QAction("Keyboard Shortcuts", self.parent)
        help_shortcuts_action.triggered.connect(self._show_shortcuts_dialog)

        help_server_log_action = QAction("Open Local Server Log", self.parent)
        help_server_log_action.triggered.connect(self._show_server_log)

        help_client_log_action = QAction("Open Local Client Log", self.parent)
        help_client_log_action.triggered.connect(self._show_client_log)

        help_menu.addAction(help_about_action)
        help_menu.addAction(help_shortcuts_action)
        help_menu.addAction(help_server_log_action)
        help_menu.addAction(help_client_log_action)

        return help_menu

    def _show_about_dialog(self):
        """Show the about dialog"""
        about_text = """
        <h1>QScope GUI</h1>
        <p>Version 0.1</p>
        <p>Developed by Sam Scholten and David Broadway </p>
        <p>For more information, visit the git repo <a href="https://github.com/DavidBroadway/Qscope">here</a> </p>
        """
        QMessageBox.about(self.parent, "About QScope", about_text)

    def _show_shortcuts_dialog(self):
        """Show dialog listing all keyboard shortcuts."""
        shortcuts_text = """                                                                                                                                     
            <h3>Keyboard Shortcuts</h3>                                                                                                                          
            <style>                                                                                                                                              
                table {                                                                                                                                          
                    border-collapse: collapse;                                                                                                                   
                    width: 100%;                                                                                                                                 
                    margin-top: 10px;                                                                                                                            
                }                                                                                                                                                
                th, td {                                                                                                                                         
                    border: 1px solid #ddd;                                                                                                                      
                    padding: 8px;                                                                                                                                
                    text-align: left;                                                                                                                            
                }                                                                                                                                                
                th {                                                                                                                                             
                    background-color: #f2f2f2;                                                                                                                   
                }                                                                                                                                                
                .section-header {                                                                                                                                
                    background-color: #e6e6e6;                                                                                                                   
                    font-weight: bold;                                                                                                                           
                }                                                                                                                                                
                .shortcut-row td {                                                                                                                               
                    padding-left: 20px;                                                                                                                          
                }                                                                                                                                                
            </style>                                                                                                                                             
            <table>                                                                                                                                              
            <tr>                                                                                                                                                 
                <th>Action</th>                                                                                                                                  
                <th>Shortcut</th>                                                                                                                                
            </tr>                                                                                                                                                
        """

        def get_shortcuts_from_widget(widget, window_name="Main"):
            """Helper function to get shortcuts from a widget and its children"""
            shortcuts = []
            # Get direct actions from the widget
            for action in widget.actions():
                if action.shortcut():
                    shortcut = action.shortcut().toString()
                    if shortcut:
                        shortcuts.append((window_name, action.text(), shortcut))

                        # If widget has children that have actions, get those too
            if hasattr(widget, "children"):
                for child in widget.children():
                    if hasattr(child, "actions"):
                        for action in child.actions():
                            if action.shortcut():
                                shortcut = action.shortcut().toString()
                                if shortcut:
                                    shortcuts.append(
                                        (window_name, action.text(), shortcut)
                                    )
            return shortcuts

        all_shortcuts = []

        # Get shortcuts from main window
        all_shortcuts.extend(get_shortcuts_from_widget(self.parent))

        # Automatically find and get shortcuts from all window attributes
        for attr_name in dir(self.parent):
            attr = getattr(self.parent, attr_name)
            # Check if attribute is a window or menu with potential shortcuts
            if isinstance(attr, (QMainWindow, QWidget, QMenu)):
                # For menu items that might have window attributes (like NotesMenu.notes_window)
                if hasattr(attr, "notes_window") and attr.notes_window:
                    all_shortcuts.extend(
                        get_shortcuts_from_widget(
                            attr.notes_window, f"{attr_name.replace('Menu', '')} Window"
                        )
                    )
                    # Get shortcuts from the menu/window itself
                all_shortcuts.extend(get_shortcuts_from_widget(attr, attr_name))

                # Sort shortcuts by window name and action name
        all_shortcuts.sort(key=lambda x: (x[0], x[1]))

        # Group shortcuts by window
        current_window = None
        for window, action, shortcut in all_shortcuts:
            if current_window != window:
                current_window = window
                shortcuts_text += f"""                                                                                                                           
                    <tr class="section-header">                                                                                                                  
                        <td colspan="2">{window}</td>                                                                                                            
                    </tr>                                                                                                                                        
                """
            shortcuts_text += f"""                                                                                                                               
                <tr class="shortcut-row">                                                                                                                        
                    <td>{action}</td>                                                                                                                            
                    <td>{shortcut}</td>                                                                                                                          
                </tr>                                                                                                                                            
            """

        shortcuts_text += "</table>"

        QMessageBox.information(self.parent, "Keyboard Shortcuts", shortcuts_text)

    def _show_server_log(self):
        try:
            # get the server log file path
            path = str(pathlib.Path.home().joinpath(".qscope/server.log"))
            webbrowser.open(path)
        except:
            pass

    def _show_client_log(self):
        try:
            # get the client log file path
            path = str(pathlib.Path.home().joinpath(".qscope/client.log"))
            webbrowser.open(path)
        except:
            pass
