from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import pyqtSignal, QPoint, Qt

class DraggableDecimalLineEdit(QLineEdit):
    focusIn = pyqtSignal()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dragging = False
        self.start_pos = QPoint()
        self.min_value = -5.0
        self.max_value = 5.0
        self.step = 0.01        

    def focusInEvent(self, event):
        self.focusIn.emit()
        super().focusInEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.start_pos = event.pos()
            try:
                self.current_value = float(self.text())
            except ValueError:
                self.current_value = 0.0
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = self.start_pos.y() - event.pos().y()
            if delta != 0:
                new_value = self.current_value + delta * self.step
                new_value = max(self.min_value, min(new_value, self.max_value))
                new_value = round(new_value * 100) / 100
                self.current_value = new_value
                self.setText(f"{new_value:.2f}")
                self.start_pos = event.pos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
        super().mouseReleaseEvent(event)
