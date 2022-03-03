from PySide2.QtCore import Qt
from PySide2.QtWidgets import QWidget, QLabel, QSlider

class SliderWidget(QWidget):
    def __init__(   self, 
                    slider_title: str, 
                    max_value: int, 
                    min_value: int, 
                    step_size: int, 
                    init_value: int=None, 
                    parent=None):
        super(SliderWidget, self).__init__(parent)

        self.title = slider_title

        if max_value < min_value:
            raise Exception('Error: Max value must be greater than min value for SliderWidget')

        self.max_value = max_value
        self.min_value = min_value
        self.step_size = step_size

        self.init_value = init_value
        if init_value is None:
            self.init_value = (max_value - min_value) / 2

        self.label_widget = QLabel(self.title + ': ' + str(init_value))
        self.label_widget.setAlignment(Qt.AlignCenter)

        self.slider_widget = QSlider(Qt.Horizontal)
        self.slider_widget.setMinimum(int(self.min_value//self.step_size))
        self.slider_widget.setMaximum(int(self.max_value//self.step_size))

        self.slider_widget.setValue(int(self.init_value/self.step_size))
        self.slider_widget.setTickPosition(QSlider.TicksBelow)
        self.slider_widget.setTickInterval(self.step_size)
    
        self.slider_widget.valueChanged.connect(self.set_slider_value_text)


    def set_slider_value_text(self):
        slider_value = self.slider_widget.value()
        self.label_widget.setText(self.title + ': ' + str(self.step_size * slider_value))


    def get_slider_label_widget(self):
        return self.label_widget


    def get_slider_widget(self):
        return self.slider_widget
    

    def get_slider_value(self):
        return self.step_size * self.slider_widget.value()
