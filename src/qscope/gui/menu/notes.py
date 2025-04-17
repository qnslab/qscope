from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFont,
    QKeySequence,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFontDialog,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from qscope.gui.util import show_warning
from qscope.gui.util.settings import GUISettings
from qscope.gui.widgets import QuComboBox

if TYPE_CHECKING:
    from qscope.gui.main_window import MainWindow


class NotesWindow(QMainWindow):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Notes")

        # Initialize settings manager
        self.settings = GUISettings(self, "NOTES")

        # Create central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Save Server Action
        save_server_action = QAction("Save on Server", self)
        save_server_action.setShortcut(QKeySequence.StandardKey.Save)
        save_server_action.setStatusTip(
            "Save notes on the server with current project name"
        )
        save_server_action.triggered.connect(self._save_notes_server)
        toolbar.addAction(save_server_action)

        # Save Local Action
        save_local_action = QAction("Save Local", self)
        save_local_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_local_action.setStatusTip("Save notes locally on this computer")
        save_local_action.triggered.connect(self._save_notes_local)
        toolbar.addAction(save_local_action)

        # Clear Action
        clear_action = QAction("Clear", self)
        clear_action.setShortcut("Ctrl+C")
        clear_action.setStatusTip("Clear all text from the notes window")
        clear_action.triggered.connect(self._clear_notes)
        toolbar.addAction(clear_action)

        # Add separator in toolbar
        toolbar.addSeparator()

        # Font chooser action
        font_action = QAction("Font...", self)
        font_action.setStatusTip("Change the font family and size")
        font_action.triggered.connect(self._choose_font)
        toolbar.addAction(font_action)

        # Font size combo box in toolbar
        self.font_size_combo = QuComboBox()
        self.font_size_combo.addItems(
            [f"{size}pt" for size in [8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24]]
        )
        self.font_size_combo.currentTextChanged.connect(
            lambda x: self._change_font_size(int(x.replace("pt", "")))
        )
        toolbar.addWidget(self.font_size_combo)

        # Create text edit with markdown highlighting
        self.text_edit = QTextEdit()
        self.text_edit.append("# Notes\n\n")

        # Set object name for settings
        self.font_size_combo.setObjectName("font_size_combo")

        self.load_font_preferences()
        self.highlighter = MarkdownHighlighter(self.text_edit.document())

        # Add zoom shortcuts
        zoom_in_action = QAction("Increase Font Size", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self._increase_font_size)
        self.addAction(zoom_in_action)

        zoom_out_action = QAction("Decrease Font Size", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self._decrease_font_size)
        self.addAction(zoom_out_action)

        layout.addWidget(self.text_edit)
        self.setCentralWidget(central_widget)
        self.resize(600, 400)

        # Add status bar
        self.statusBar().show()

    def _save_notes_server(self):
        """Save notes using connection manager"""
        if not self.parent.connection_manager.is_connected():
            QMessageBox.warning(self, "Warning", "Not connected to server")
            return

        try:
            # Get project name from status bar
            project_name = self.parent.get_project_name()
            if not project_name:
                show_warning("Please set a project name in the status bar")
                return

                # Get current notes text
            notes_text = self.text_edit.toPlainText()

            if not notes_text.strip():
                show_warning("No notes to save")
                return

                # Save notes through connection manager
            self.parent.connection_manager.save_notes(project_name, notes_text)
            self.statusBar().showMessage("Notes saved on server successfully", 3000)
            logger.info("Notes saved on server successfully")

        except Exception as e:
            error_msg = f"Error saving notes: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            logger.error(error_msg)

    def _save_notes_local(self):
        """Save notes locally with file chooser"""
        notes_text = self.text_edit.toPlainText()

        if not notes_text.strip():
            QMessageBox.warning(self, "Warning", "No notes to save")
            return

        try:
            # Open file dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Notes Locally",
                "",
                "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)",
            )

            if file_path:
                with open(file_path, "w") as f:
                    f.write(notes_text)
                self.statusBar().showMessage(
                    f"Notes saved locally to {file_path}", 3000
                )
                logger.info(f"Notes saved locally to {file_path}")

        except Exception as e:
            error_msg = f"Error saving notes locally: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            logger.error(error_msg)

    def _clear_notes(self):
        """Clear the text edit after confirmation"""
        if self.text_edit.toPlainText().strip():
            reply = QMessageBox.question(
                self,
                "Clear Notes",
                "Are you sure you want to clear all notes?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.text_edit.clear()
                self.statusBar().showMessage("Notes cleared", 3000)
                logger.info("Notes cleared")

    def get_notes(self) -> str:
        return self.text_edit.toPlainText()

    def has_notes(self) -> bool:
        return bool(self.text_edit.toPlainText().strip())

    def _choose_font(self):
        """Open font dialog and update font"""
        current = self.text_edit.font()
        font, ok = QFontDialog.getFont(current, self)
        if ok:
            self.text_edit.setFont(font)
            self.save_font_preferences()
            self.statusBar().showMessage(
                f"Font updated to {font.family()} {font.pointSize()}pt", 3000
            )

    def _change_font_size(self, size):
        """Change the font size of the text edit"""
        font = self.text_edit.font()
        font.setPointSize(size)
        self.text_edit.setFont(font)
        self.save_font_preferences()
        self.statusBar().showMessage(f"Font size changed to {size}pt", 3000)

    def _increase_font_size(self):
        """Increase font size by 1pt"""
        font = self.text_edit.font()
        current_size = font.pointSize()
        new_size = current_size + 1
        font.setPointSize(new_size)
        self.text_edit.setFont(font)
        self.save_font_preferences()
        # Update combo box
        size_idx = self.font_size_combo.findText(f"{new_size}pt")
        if size_idx >= 0:
            self.font_size_combo.setCurrentIndex(size_idx)
        self.statusBar().showMessage(f"Font size increased to {new_size}pt", 3000)

    def _decrease_font_size(self):
        """Decrease font size by 1pt"""
        font = self.text_edit.font()
        current_size = font.pointSize()
        if current_size > 6:  # Prevent too small fonts
            new_size = current_size - 1
            font.setPointSize(new_size)
            self.text_edit.setFont(font)
            self.save_font_preferences()
            # Update combo box
            size_idx = self.font_size_combo.findText(f"{new_size}pt")
            if size_idx >= 0:
                self.font_size_combo.setCurrentIndex(size_idx)
            self.statusBar().showMessage(f"Font size decreased to {new_size}pt", 3000)

    def save_font_preferences(self):
        """Save current font preferences"""
        font = self.text_edit.font()
        config = self.settings.make_config()
        config = self.settings.add_section(config, "NOTES")
        config["NOTES"]["font_family"] = font.family()
        config["NOTES"]["font_size"] = str(font.pointSize())
        self.settings.save_config(config)

    def load_font_preferences(self):
        """Load saved font preferences or use defaults"""
        # Create default settings if they don't exist
        if not self.settings.prev_state_ini.exists():
            config = self.settings.make_config()
            config["NOTES"] = {"font_family": "Courier New", "font_size": "10"}
            self.settings.save_config(config)

        self.settings.config.read(self.settings.prev_state_ini)
        try:
            family = self.settings.config.get(
                "NOTES", "font_family", fallback="Courier New"
            )
            size = self.settings.config.getint("NOTES", "font_size", fallback=10)
        except Exception as e:
            logger.error(f"Error loading font preferences: {e}")
            family = "Courier New"
            size = 10

        font = QFont(family, size)
        self.text_edit.setFont(font)

        # Update font size combo box
        size_idx = self.font_size_combo.findText(f"{size}pt")
        if size_idx >= 0:
            self.font_size_combo.setCurrentIndex(size_idx)

    def get_notes(self) -> str:
        return self.text_edit.toPlainText()


class NotesMenu(QMenu):
    def __init__(self, parent: MainWindow, menuBar: QMenuBar):
        super().__init__()
        self.parent = parent
        self._create_notes_menu(menuBar)
        self.notes_window = None

    def _create_notes_menu(self, menuBar):
        notes_menu = menuBar.addMenu("&Notes")

        open_notes_action = QAction("&Open Notes", self.parent)
        open_notes_action.triggered.connect(self._open_notes)

        notes_menu.addAction(open_notes_action)
        return notes_menu

    def _open_notes(self):
        if self.notes_window is None:
            self.notes_window = NotesWindow(self.parent)
        self.notes_window.show()

    def get_notes(self) -> str:
        # Return notes from notes window only if it is open
        if self.notes_window is not None and self.notes_window.isVisible():
            return self.notes_window.get_notes()
        return ""


class MarkdownHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.highlighting_rules = []

        # Header format
        header_format = QTextCharFormat()
        header_format.setForeground(QColor("#59b6e6"))  # Blue
        header_format.setFontWeight(QFont.Weight.Bold)
        pattern = QRegularExpression("^(#+)\\s.*$")
        self.highlighting_rules.append((pattern, header_format))

        # Bold format (using double stars)
        bold_format = QTextCharFormat()
        bold_format.setForeground(QColor("#FF0000"))  # Red
        bold_format.setFontWeight(QFont.Weight.Bold)
        pattern = QRegularExpression(r"\*\*(.*?)\*\*")
        self.highlighting_rules.append((pattern, bold_format))

        # Bold format (using double underscores)
        bold_format_underscore = QTextCharFormat()
        bold_format_underscore.setForeground(QColor("#FF0000"))  # Red
        bold_format_underscore.setFontWeight(QFont.Weight.Bold)
        pattern = QRegularExpression(r"__(.*?)__")
        self.highlighting_rules.append((pattern, bold_format_underscore))

        # Italic format (using single stars)
        italic_format = QTextCharFormat()
        italic_format.setForeground(QColor("#008000"))  # Green
        italic_format.setFontItalic(True)
        pattern = QRegularExpression(r"\*(.*?)\*")
        self.highlighting_rules.append((pattern, italic_format))

        # Italic format (using single underscores)
        italic_format_underscore = QTextCharFormat()
        italic_format_underscore.setForeground(QColor("#008000"))  # Green
        italic_format_underscore.setFontItalic(True)
        pattern = QRegularExpression(r"_(.*?)_")
        self.highlighting_rules.append((pattern, italic_format_underscore))

        # Code format
        code_format = QTextCharFormat()
        code_format.setForeground(QColor("#A52A2A"))  # Brown
        code_format.setFontFamily("Courier")
        pattern = QRegularExpression("`([^`]*)`")
        self.highlighting_rules.append((pattern, code_format))

        # Link format
        link_format = QTextCharFormat()
        link_format.setForeground(QColor("#59b6e6"))  # Blue
        pattern = QRegularExpression(r"\[.*?\]\(.*?\)")
        self.highlighting_rules.append((pattern, link_format))

    def highlightBlock(self, text):
        for pattern, fmt in self.highlighting_rules:
            match = pattern.globalMatch(text)
            while match.hasNext():
                m = match.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)
