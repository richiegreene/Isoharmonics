from PyQt5.QtWidgets import QWidget, QMainWindow, QSplitter, QVBoxLayout, QPushButton, QLabel, QLineEdit, QHBoxLayout, QToolButton, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from fractions import Fraction
from gui.widgets.isohe_widget import IsoHEWidget
from theory.triangle_generator import generate_triangle_image

class IsoHEWindow(QMainWindow):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.setWindowTitle("Triadic Harmonic Entropy")
        self.setGeometry(150, 150, 600, 550)
        self.setStyleSheet("background-color: #23262F;")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.sidebar_width = 200

        # Load custom font
        self.custom_font = QFont("Arial Nova", 12)

        self.splitter = QSplitter()
        self.setCentralWidget(self.splitter)

        self.sidebar = QWidget()
        self.sidebar.setStyleSheet("background-color: #23262F;")
        self.sidebar.setMinimumWidth(0)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(9, 9, 9, 9)
        self.sidebar_layout.setSpacing(4)

        self.collapse_button = QToolButton()
        self.collapse_button.setStyleSheet(self.button_style())
        self.collapse_button.setArrowType(Qt.RightArrow)
        self.collapse_button.clicked.connect(self.toggle_sidebar)
        self.sidebar_layout.addWidget(self.collapse_button)
        
        # Generate Triangle
        self.generate_button = QPushButton("Generate Triangle")
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #2C2F3B;
                color: white;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 4px;
            }
        """)
        self.generate_button.clicked.connect(self.generate_triangle_image)
        self.sidebar_layout.addWidget(self.generate_button)
        
        # Equave input, compact format
        eq_layout = QVBoxLayout()
        equave_label = QLabel() # Removed 'Equave' header
        equave_label.setStyleSheet("color: white;")
        equave_label.setAlignment(Qt.AlignLeft)
        eq_layout.addWidget(equave_label)

        eq_row = QHBoxLayout()
        eq_row.setSpacing(2)
        eq_row.setContentsMargins(0, 0, 0, 0)
        self.equave_start = QLineEdit("1")
        self.equave_end = QLineEdit("2")
        for field in [self.equave_start, self.equave_end]:
            field.setStyleSheet("background-color: #2C2F3B; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px; padding-left: 6px;")
            field.setMaximumWidth(32)
            field.setFixedHeight(22)
            field.textChanged.connect(self.update_equave)
        eq_row.addWidget(self.equave_start)
        to_label_eq = QLabel(":")
        to_label_eq.setStyleSheet("color: white;")
        eq_row.addWidget(to_label_eq)
        eq_row.addWidget(self.equave_end)
        eq_row.addStretch(1)
        eq_layout.addLayout(eq_row)
        self.sidebar_layout.addLayout(eq_layout)

        # Create isohe_widget
        self.isohe_widget = IsoHEWidget(main_app)
        
        # Cent display and pivot buttons
        self.pivot_buttons = {}
        self.cent_labels = {}
        self.note_labels = {}
        for pivot_name in ["3", "2", "1"]:
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0,0,0,0)
            row_layout.setSpacing(0)

            button = QPushButton(pivot_name)
            button.setStyleSheet(self.pivot_button_style())
            button.setCheckable(True)
            button.setFixedSize(20, 20)
            button.clicked.connect(lambda checked, name=pivot_name: self.set_pivot(name))
            self.pivot_buttons[pivot_name] = button
            row_layout.addWidget(button, 0, Qt.AlignLeft)

            spacer1 = QSpacerItem(10, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)
            row_layout.addItem(spacer1)

            label = QLabel("0")
            label.setStyleSheet("color: white; font-size: 12pt;")
            self.cent_labels[pivot_name] = label
            row_layout.addWidget(label, 0, Qt.AlignLeft)

            spacer2 = QSpacerItem(10, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)
            row_layout.addItem(spacer2)

            note_label = QLabel("C₄")
            note_label.setFont(self.custom_font)
            note_label.setStyleSheet("color: white;")
            self.note_labels[pivot_name] = note_label
            row_layout.addWidget(note_label, 1, Qt.AlignLeft)

            self.sidebar_layout.addLayout(row_layout)

        self.isohe_widget.cents_label = self.cent_labels
        self.isohe_widget.note_labels = self.note_labels
        
        # Add ratios label below cents
        self.isohe_widget.ratios_label = QLabel()
        self.isohe_widget.ratios_label.setStyleSheet("color: white; font-size: 12pt;")
        self.isohe_widget.ratios_label.setFixedHeight(28)
        self.isohe_widget.ratios_label.setWordWrap(True)
        self.isohe_widget.ratios_label.setAlignment(Qt.AlignLeft)
        self.sidebar_layout.addWidget(self.isohe_widget.ratios_label)
        self.sidebar_layout.addStretch()

        self.expand_button = QToolButton(self)
        self.expand_button.setStyleSheet(self.button_style())
        self.expand_button.setArrowType(Qt.LeftArrow)
        self.expand_button.setFixedSize(30, 30)
        self.expand_button.clicked.connect(self.toggle_sidebar)
        self.expand_button.show()

        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.isohe_widget)
        self.splitter.setSizes([0, self.width()])
        
        self.current_pivot = "1"  # Initialize default current_pivot
        self.pivot_buttons[self.current_pivot].setChecked(True) # Set initial checked state

        self.update_equave()
        self.collapse_sidebar()

    def mousePressEvent(self, event):
        focused_widget = self.focusWidget()
        if isinstance(focused_widget, QLineEdit):
            focused_widget.clearFocus()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_1:
            self.set_pivot("1")
        elif key == Qt.Key_2:
            self.set_pivot("2")
        elif key == Qt.Key_3:
            self.set_pivot("3")
        else:
            super().keyPressEvent(event)

    def button_style(self):
        return """
            QToolButton {
                background: #343744;
                border: none;
                padding: 5px;
            }
            QToolButton:hover {
                background: #404552;
            }
        """

    def pivot_button_style(self):
        return """
            QPushButton {
                background-color: #2C2F3B;
                color: white;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #0437f2; 
                color: white;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 4px;
            }
        """

    def set_pivot(self, pivot_name):
        if self.current_pivot != pivot_name:
            self.pivot_buttons[self.current_pivot].setChecked(False)
            self.current_pivot = pivot_name
            self.pivot_buttons[pivot_name].setChecked(True)
            
            pivot_map = {"3": "upper", "2": "middle", "1": "lower"}
            self.isohe_widget.set_pivot_voice(pivot_map[pivot_name])

    def toggle_sidebar(self):
        if self.sidebar.width() == 0:
            self.expand_sidebar()
        else:
            self.collapse_sidebar()

    def expand_sidebar(self):
        self.splitter.setSizes([self.sidebar_width, self.width() - self.sidebar_width])
        self.expand_button.hide()

    def collapse_sidebar(self):
        self.splitter.setSizes([0, self.width()])
        self.expand_button.show()

    def resizeEvent(self, event):
        self.expand_button.move(10, 10)
        super().resizeEvent(event)
        self.isohe_widget.update_dimensions()

    def update_equave(self):
        try:
            row1_idx = int(self.equave_start.text())
            row2_idx = int(self.equave_end.text())

            table = self.main_app.table
            if not (0 <= row1_idx < table.rowCount() and 0 <= row2_idx < table.rowCount()):
                return

            item1 = table.item(row1_idx, 1)
            item2 = table.item(row2_idx, 1)

            if item1 and item2:
                ratio1 = Fraction(item1.text())
                ratio2 = Fraction(item2.text())
                
                if ratio1 == 0: return
                
                equave_ratio = ratio2 / ratio1
                self.isohe_widget.set_equave(equave_ratio)

        except (ValueError, ZeroDivisionError):
            pass # Ignore invalid numbers for now

    def closeEvent(self, event):
        self.isohe_widget.stop_sound()
        super().closeEvent(event)

    def generate_triangle_image(self):
        if hasattr(self.isohe_widget, 'equave'):
            image = generate_triangle_image(
                self.isohe_widget.equave,
                self.isohe_widget.width(),
                self.isohe_widget.height()
            )
            if image:
                self.isohe_widget.set_triangle_image(image)
