from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import pyqtSignal, QPoint, Qt
from fractions import Fraction
from math import gcd

class DraggableFractionLineEdit(QLineEdit):
    focusIn = pyqtSignal()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dragging = False
        self.start_pos = QPoint()
        self.current_ratio = Fraction("1/1")

    def focusInEvent(self, event):
        self.focusIn.emit()
        super().focusInEvent(event)    

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.start_pos = event.pos()
            try:
                self.current_ratio = Fraction(self.text())
            except ValueError:
                self.current_ratio = Fraction("1/1")
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = self.start_pos.y() - event.pos().y()
            if delta != 0:
                self.adjust_ratio(delta)
                self.start_pos = event.pos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
        super().mouseReleaseEvent(event)

    def adjust_ratio(self, delta):
        intervals = self.generate_95_odd_limit_intervals()
        try:
            current_index = intervals.index(self.current_ratio)
        except ValueError:
            current_index = 0
        new_index = current_index + delta
        if new_index < 0:
            new_index = 0
        elif new_index >= len(intervals):
            new_index = len(intervals) - 1
        self.current_ratio = intervals[new_index]
        self.setText(str(self.current_ratio))

    def generate_95_odd_limit_intervals(self):
        intervals = []
        for numerator in range(1, 96):
            for denominator in range(1, 96):
                if gcd(numerator, denominator) == 1:
                    ratio = Fraction(numerator, denominator)
                    if ratio >= Fraction(1, 16) and ratio <= Fraction(16, 1):
                        intervals.append(ratio)
        intervals.sort(key=lambda x: float(x))
        return intervals
