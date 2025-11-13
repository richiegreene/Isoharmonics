import sys
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QDockWidget, QSpinBox,
                             QMessageBox, QProgressDialog, QComboBox)
from PyQt5.QtCore import Qt
from tetrahedron_generator import generate_tetrahedron_data, generate_odd_limit_points
from widgets.tetrahedron_widget import TetrahedronWidget

class FourHEWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("4-Note Harmonic Entropy Visualizer")
        self.setGeometry(100, 100, 1000, 800)

        self.tetra_widget = TetrahedronWidget()
        self.setCentralWidget(self.tetra_widget)

        self.create_control_panel()
        self.update_visualization()

    def create_control_panel(self):
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)

        # Equave Ratio
        equave_layout = QHBoxLayout()
        equave_label = QLabel("Equave Ratio:")
        self.equave_input = QLineEdit("2.0")
        equave_layout.addWidget(equave_label)
        equave_layout.addWidget(self.equave_input)
        control_layout.addLayout(equave_layout)

        # Resolution
        res_layout = QHBoxLayout()
        res_label = QLabel("Resolution:")
        self.resolution_input = QSpinBox()
        self.resolution_input.setRange(10, 300)
        self.resolution_input.setValue(60)
        self.resolution_input.setToolTip("Higher values increase detail but require much more computation time.")
        res_layout.addWidget(res_label)
        res_layout.addWidget(self.resolution_input)
        control_layout.addLayout(res_layout)

        # View Mode Dropdown
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Volume Data", "Scatter Plot"])
        control_layout.addWidget(self.view_mode_combo)

        # Odd Limit
        self.odd_limit_layout_widget = QWidget()
        self.odd_limit_layout = QHBoxLayout(self.odd_limit_layout_widget)
        self.odd_limit_layout.setContentsMargins(0,0,0,0)
        odd_limit_label = QLabel("Odd-Limit:")
        self.odd_limit_input = QLineEdit("9")
        self.odd_limit_layout.addWidget(odd_limit_label)
        self.odd_limit_layout.addWidget(self.odd_limit_input)
        control_layout.addWidget(self.odd_limit_layout_widget)

        # Toggle visibility of odd-limit based on view mode
        self.view_mode_combo.currentTextChanged.connect(self.toggle_odd_limit_visibility)
        self.toggle_odd_limit_visibility("Volume Data") # Initial state

        # Update Button
        self.update_button = QPushButton("Update Visualization")
        self.update_button.clicked.connect(self.update_visualization)
        control_layout.addWidget(self.update_button)

        control_layout.addStretch()

        dock_widget = QDockWidget("Controls", self)
        dock_widget.setWidget(control_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_widget)

    def toggle_odd_limit_visibility(self, view_mode):
        """Shows or hides the Odd-Limit input based on the selected view mode."""
        is_scatter = (view_mode == "Scatter Plot")
        self.odd_limit_layout_widget.setVisible(is_scatter)

    def update_visualization(self):
        try:
            equave_ratio = float(self.equave_input.text())
            if equave_ratio <= 1:
                raise ValueError("Equave must be a positive number > 1.")
            
            view_mode = self.view_mode_combo.currentText()
            show_volume = (view_mode == "Volume Data")
            show_points = (view_mode == "Scatter Plot")
            
            odd_limit = 0
            if show_points:
                odd_limit = int(self.odd_limit_input.text())
                if odd_limit < 3 or odd_limit % 2 == 0:
                    raise ValueError("Odd-Limit must be an odd number >= 3.")

        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            return

        resolution = self.resolution_input.value()
        
        progress = QProgressDialog("Generating Visualization Data...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()

        volume_data = None
        if show_volume or show_points:
            volume_data = generate_tetrahedron_data(equave_ratio, resolution)
            if volume_data is None or volume_data[3] is None:
                QMessageBox.warning(self, "Generation Error", "Could not generate HE data for coordinate system.")
                progress.close()
                return
        
        progress.setValue(50)
        QApplication.processEvents()

        points_data = None
        if show_points:
            points_data = generate_odd_limit_points(odd_limit, equave_ratio)

        self.tetra_widget.update_tetrahedron(
            volume_data=volume_data, 
            points_data=points_data,
            show_volume=show_volume,
            show_points=show_points
        )
        
        progress.setValue(100)
        progress.close()