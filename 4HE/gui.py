import sys
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QDockWidget, QSpinBox,
                             QMessageBox, QProgressDialog)
from PyQt5.QtCore import Qt
from tetrahedron_generator import generate_tetrahedron_data
from widgets.tetrahedron_widget import TetrahedronWidget

class FourHEWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("4-Note Harmonic Entropy Visualizer")
        self.setGeometry(100, 100, 1000, 800)

        # The widget will be a pyqtgraph GLViewWidget, which is a QWidget
        self.tetra_widget = TetrahedronWidget()
        self.setCentralWidget(self.tetra_widget)

        self.create_control_panel()

        # Initial generation
        self.update_visualization()

    def create_control_panel(self):
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)

        # Ratios
        # Ratio inputs removed to focus on HE visualization
        self.ratio_inputs = []
        
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

        # Update Button
        self.update_button = QPushButton("Update Visualization")
        self.update_button.clicked.connect(self.update_visualization)
        control_layout.addWidget(self.update_button)

        control_layout.addStretch()

        dock_widget = QDockWidget("Controls", self)
        dock_widget.setWidget(control_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_widget)

    def update_visualization(self):
        try:
            equave_ratio = float(self.equave_input.text())
            if equave_ratio <= 1:
                raise ValueError("Equave must be a positive number > 1.")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            return

        resolution = self.resolution_input.value()

        progress = QProgressDialog("Generating HE data...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()

        # Call the new generator function
        data = generate_tetrahedron_data(equave_ratio, resolution)
        
        if data is None or data[0] is None:
            QMessageBox.warning(self, "Generation Error", "Could not generate data. Try different limits.")
            progress.close()
            return

        progress.setValue(50)
        QApplication.processEvents()

        # Pass data to the widget (no JI points)
        self.tetra_widget.update_tetrahedron(data, None)
        
        progress.setValue(100)
        progress.close()