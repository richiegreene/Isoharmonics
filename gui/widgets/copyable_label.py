from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import Qt

class CopyableLabel(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("""
            QLineEdit {
                border: none;
                background: transparent;
                color: #FFFFFF;
                font: inherit;
                text-align: center;
            }
        """)
        self.setReadOnly(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.IBeamCursor)
        self.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.setAttribute(Qt.WA_MacShowFocusRect, False)
