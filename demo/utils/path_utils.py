from pathlib import Path, PureWindowsPath, PurePosixPath
import platform

#---------------------------------------------------------------
def native_path_string(_path: Path):
    if platform.system() == 'Darwin':
        return str(PurePosixPath(_path))
    elif platform.system() == 'Windows':
        return str(PureWindowsPath(_path))
    elif platform.system() == 'Linux':
        return str(PurePosixPath(_path))