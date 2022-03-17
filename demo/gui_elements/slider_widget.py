from PySide2.QtCore import Qt
from PySide2.QtWidgets import QWidget, QLabel, QSlider

from typing import Callable

class SliderWidget(QWidget):
    def __init__(   self, 
                    slider_title: str, 
                    max_value: int, 
                    min_value: int, 
                    step_size: int, 
                    init_value: int=None,
                    vertical: bool=False,
                    position_idx: int=None,
                    on_change_callback: Callable=None,
                    label_precision: int=2,
                    parent=None):
        super(SliderWidget, self).__init__(parent)

        self.title = slider_title

        if max_value < min_value:
            raise Exception('Error: Max value must be greater than min value for SliderWidget')

        self.max_value = max_value
        self.min_value = min_value
        self.step_size = step_size
        self.position_idx = position_idx

        self.init_value = init_value
        if init_value is None:
            self.init_value = (max_value - min_value) / 2

        self.label_widget = QLabel(self.title + ': ' + str(init_value))
        self.label_widget.setAlignment(Qt.AlignCenter)

        self.slider_widget = QSlider(Qt.Vertical) if vertical else QSlider(Qt.Horizontal)
        self.slider_widget.setMinimum(int(self.min_value//self.step_size))
        self.slider_widget.setMaximum(int(self.max_value//self.step_size))

        self.slider_widget.setValue(int(self.init_value/self.step_size))
        self.slider_widget.setTickPosition(QSlider.TicksLeft) if vertical else self.slider_widget.setTickPosition(QSlider.TicksBelow)
        self.slider_widget.setTickInterval(self.step_size)
    
        self.slider_widget.valueChanged.connect(self.set_slider_value_text)
        self.on_change_callback = on_change_callback

        self.label_precision = label_precision


    def set_slider_value_text(self):
        slider_value = self.slider_widget.value()
        self.label_widget.setText(self.title + ': ' + f"{self.step_size * slider_value:.{self.label_precision}f}")
        if self.on_change_callback is not None: self.on_change_callback()

    def set_slider_title(self, title):
        self.title = title
        self.set_slider_value_text()

    def get_slider_label_widget(self):
        return self.label_widget

    def get_slider_widget(self):
        return self.slider_widget
    
    def get_slider_value(self):
        return self.step_size * self.slider_widget.value()

    def get_position_idx(self):
        return self.position_idx

