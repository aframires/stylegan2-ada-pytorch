from PySide2.QtWidgets import QFrame

class HorLineWidget(QFrame):
    def __init__(self):
        super(HorLineWidget, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class VertLineWidget(QFrame):
    def __init__(self):
        super(VertLineWidget, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)