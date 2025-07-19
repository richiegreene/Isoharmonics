from PyQt5.QtWidgets import QWidget, QMainWindow, QSplitter, QVBoxLayout, QToolButton, QPushButton, QLabel, QLineEdit, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
from gui.widgets.lattice_widget import LatticeWidget

class LatticeWindow(QMainWindow):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.setWindowTitle("Isoharmonic Lattice")
        self.setGeometry(100, 100, 700, 500)
        self.setStyleSheet("background-color: #23262F;")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.sidebar_width = 200
        self.is_edo_mode = False

        # Main splitter layout
        self.splitter = QSplitter()
        self.setCentralWidget(self.splitter)

        # Sidebar
        self.sidebar = QWidget()
        self.sidebar.setStyleSheet("background-color: #23262F;")
        self.sidebar.setMinimumWidth(0)
        self.sidebar_layout = QVBoxLayout(self.sidebar)

        # Collapse button
        self.collapse_button = QToolButton()
        self.collapse_button.setStyleSheet(self.button_style())
        self.collapse_button.setArrowType(Qt.RightArrow)
        self.collapse_button.clicked.connect(self.toggle_sidebar)
        self.sidebar_layout.addWidget(self.collapse_button)

        # JI/EDO Toggle Button
        self.ji_edo_toggle = QPushButton("JI")
        self.ji_edo_toggle.setCheckable(True)
        self.ji_edo_toggle.setStyleSheet("""
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
        """)
        self.ji_edo_toggle.toggled.connect(self.toggle_ji_edo)
        self.sidebar_layout.addWidget(self.ji_edo_toggle)

        # Equave Toggle Button
        self.equave_toggle = QPushButton("Equave Reduction")
        self.equave_toggle.setCheckable(True)
        self.equave_toggle.setStyleSheet("""
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
        """)
        self.equave_toggle.toggled.connect(self.update_lattice)
        self.sidebar_layout.addWidget(self.equave_toggle)

        # Lattice Distance Inputs
        header_label = QLabel("Interval Between Partials", self)
        header_label.setStyleSheet("color: white;")
        self.sidebar_layout.addWidget(header_label)
        arrow_style = "color: white; font-weight: bold;"
        entry_style = """
            QLineEdit {
                background-color: #2C2F3B;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 2px;
            }
        """
    
        # Equave input: [Equave] [x] : [y]
        eq_layout = QHBoxLayout()
        equave_label = QLabel("Equave")
        equave_label.setStyleSheet("color: white;")
        eq_layout.addWidget(equave_label)

        self.equave_start = QLineEdit("1")
        self.equave_end = QLineEdit("2")
        for field in [self.equave_start, self.equave_end]:
            field.setStyleSheet("background-color: #2C2F3B; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px;")
            field.textChanged.connect(self.update_lattice)
        eq_layout.addWidget(self.equave_start)

        to_label_eq = QLabel(":")
        to_label_eq.setStyleSheet("color: white;")
        eq_layout.addWidget(to_label_eq)

        eq_layout.addWidget(self.equave_end)
        self.sidebar_layout.addLayout(eq_layout)

        # → Horizontal axis
        h_layout = QHBoxLayout()
        h_arrow = QLabel("     →     ")
        h_arrow.setStyleSheet(arrow_style)
        self.h_start = QLineEdit("2")
        self.h_end = QLineEdit("3")
        for field in [self.h_start, self.h_end]:
            field.setStyleSheet(entry_style)
            field.textChanged.connect(self.update_lattice)
        h_layout.addWidget(h_arrow)
        h_layout.addWidget(self.h_start)
        
        to_label_h = QLabel(":")
        to_label_h.setStyleSheet("color: white;")
        h_layout.addWidget(to_label_h)

        h_layout.addWidget(self.h_end)
        self.sidebar_layout.addLayout(h_layout)

        # ↗ Diagonal axis
        d_layout = QHBoxLayout()
        d_arrow = QLabel("     ↗     ")
        d_arrow.setStyleSheet(arrow_style)
        self.d_start = QLineEdit("4")
        self.d_end = QLineEdit("5")
        for field in [self.d_start, self.d_end]:
            field.setStyleSheet(entry_style)
            field.textChanged.connect(self.update_lattice)
        d_layout.addWidget(d_arrow)
        d_layout.addWidget(self.d_start)
        
        to_label_d = QLabel(":")
        to_label_d.setStyleSheet("color: white;")
        d_layout.addWidget(to_label_d)

        d_layout.addWidget(self.d_end)
        self.sidebar_layout.addLayout(d_layout)

        # Style + connect
        for field in [self.h_start, self.h_end, self.d_start, self.d_end]:
            field.setStyleSheet("background-color: #2C2F3B; color: white; border: 1px solid #444;")
            field.textChanged.connect(self.update_lattice)

        entry_fields = [self.h_start, self.h_end, self.d_start, self.d_end]
        for field in entry_fields:
            field.setStyleSheet("""
                QLineEdit {
                    background-color: #2C2F3B;
                    color: white;
                    border: 1px solid #444;
                    border-radius: 4px;
                    padding: 2px;
                }
            """)    

        self.sidebar_layout.addStretch()

        # Expand button (floating)
        self.expand_button = QToolButton(self)
        self.expand_button.setStyleSheet(self.button_style())
        self.expand_button.setArrowType(Qt.LeftArrow)
        self.expand_button.setFixedSize(30, 30)
        self.expand_button.clicked.connect(self.toggle_sidebar)
        self.expand_button.show()

        # Lattice widget
        self.lattice_widget = LatticeWidget(main_app)
        self.lattice_widget.setStyleSheet("background-color: #23262F;")

        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.lattice_widget)
        
        self.update_lattice()
        self.collapse_sidebar()  # Start collapsed

        # Grid update triggers
        self.main_app.fundamental_entry.textChanged.connect(self.lattice_widget.update_grid)
        self.main_app.isoharmonic_entry.textChanged.connect(self.lattice_widget.update_grid)
        self.main_app.partials_above_entry.textChanged.connect(self.lattice_widget.update_grid)
        self.main_app.partials_below_entry.textChanged.connect(self.lattice_widget.update_grid)
        
        QTimer.singleShot(0, self.update_lattice)

    def toggle_ji_edo(self, checked):
        self.is_edo_mode = checked
        self.ji_edo_toggle.setText("EDO" if checked else "JI")
        self.update_lattice()

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

    def update_lattice(self):
        self.lattice_widget.update_grid()

    def closeEvent(self, event):
        if hasattr(self.lattice_widget, 'current_sound') and self.lattice_widget.current_sound is not None:
            self.lattice_widget.current_sound.stop()
        super().closeEvent(event)
        self.main_app.lattice_window = None
