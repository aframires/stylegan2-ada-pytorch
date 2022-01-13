from pathlib import Path
from PySide2.QtWidgets import QWidget, QFileDialog

from utils.path_utils import native_path_string

class FileDialogWidget(QWidget):
    def __init__(self, title: str, file_type_filter: str=None, default_filter: str=None):
        super(FileDialogWidget, self).__init__()

        self.title = title
        self.file_type_filter = f"All Files (*);;{file_type_filter}"
        self.default_filter = default_filter

        self.left = 10
        self.top = 10

        self.width = 640
        self.height = 480

        self.file_parent_path = None

        self.dialog = QFileDialog()
        self.dialog.setWindowTitle(self.title)

        self.__init_UI()


    def __init_UI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)


    def choose_open_file_path(self) -> Path:
        if self.file_parent_path is not None:
            self.dialog.setDirectory(native_path_string(self.file_parent_path))

        full_filepath_str, _ = self.dialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "", self.file_type_filter, self.default_filter)

        if full_filepath_str:
            full_file_path = Path(full_filepath_str)
            self.file_parent_path = full_file_path.parent

            return full_file_path

        return None


    def choose_save_file_path(self) -> Path:
        if self.file_parent_path is not None:
            self.dialog.setDirectory(native_path_string(self.file_parent_path))

        full_filepath_str, _ = self.dialog.getSaveFileName(self,"QFileDialog.getOpenFileName()", "", self.file_type_filter)

        if full_filepath_str:
            full_file_path = Path(full_filepath_str)
            self.file_parent_path = full_file_path.parent

            return full_file_path

        return None
