from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import pyqtSignal, QPoint, Qt

class DraggableIntegerLineEdit(QLineEdit):
    focusIn = pyqtSignal()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dragging = False
        self.start_pos = QPoint()
        self.min_value = 0
        self.max_value = 100
    def focusInEvent(self, event):
        self.focusIn.emit()
        super().focusInEvent(event)        

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = self.start_pos.y() - event.pos().y()
            if delta != 0:
                self.adjust_value(delta)
                self.start_pos = event.pos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
        super().mouseReleaseEvent(event)

    def adjust_value(self, delta):
        try:
            current_value = int(self.text())
        except ValueError:
            current_value = 0
        new_value = current_value + delta
        if new_value < self.min_value:
            new_value = self.min_value
        elif new_value > self.max_value:
            new_value = self.max_value
        self.setText(str(new_value))

    def set_constraints(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

class ReversedDraggableIntegerLineEdit(DraggableIntegerLineEdit):
    def adjust_value(self, delta):
        try:
            current_value = int(self.text())
        except ValueError:
            current_value = 0
        new_value = current_value - delta
        if new_value < self.min_value:
            new_value = self.min_value
        elif new_value > self.max_value:
            new_value = self.max_value
        self.setText(str(new_value))
