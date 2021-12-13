from PySide2.QtWidgets import QLabel
from PySide2.QtGui import QPixmap, QColor, QPainter

class CanvasWidget(QLabel):
    def __init__(self, width, height, pixel_width=3, data_update_fctr=None, sync_fctr=None):
        super().__init__()
        self.pixel_width = pixel_width
        self.canvas_width = width
        self.canvas_height = height * self.pixel_width
        pixmap = QPixmap(self.canvas_width, self.canvas_height)
        self.setPixmap(pixmap)

        self.data_update_fctr = data_update_fctr
        self.sync_fctr = sync_fctr

        self.last_x, self.last_y = None, None
        self.pen_color = QColor('#FDFE02')

        self.clear()

    # overrides base class method (don't change to snake case!)
    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        if abs(e.x()) >= self.canvas_width:
            return

        if e.x() == self.last_x:
            self.last_y = e.y()
            return


        if not abs(e.x() - self.last_x) == 1 and not (e.y() - self.last_y):

            num_steps_x = abs(e.x() - self.last_x)
            step_size_y = abs(e.y() - self.last_y) / num_steps_x

            y_val = e.y()
            if e.x() > self.last_x:
                # moving left to right
                for idx in range(self.last_x, e.x()):
                    self.data_update_fctr(idx, y_val/self.canvas_height - 0.5)
                    y_val += step_size_y
            else:
                # moving right to left
                for idx in range(e.x(), self.last_x):
                    self.data_update_fctr(idx, y_val/self.canvas_height - 0.5)
                    y_val += step_size_y

        else:
            self.data_update_fctr(e.x(), e.y()/self.canvas_height - 0.5)


        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

        self.sync_fctr()

    # overrides base class method (don't change to snake case!)
    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None
    

    def paint_data(self, data):
        painter = QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(self.pixel_width)
        p.setColor(self.pen_color)
        painter.setPen(p)

        for idx, val in enumerate(data.squeeze()):
            
            # transform data [-1.0, 1.0] --> [0, height]
            val = int(self.canvas_height * (val + 0.5))
            painter.drawPoint(idx, val)
        
        painter.end()
        self.update()


    def clear(self):
        self.pixmap().fill(QColor('#0E1111'))