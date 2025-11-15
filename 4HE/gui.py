import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QDockWidget, QSpinBox,
                             QMessageBox, QProgressDialog, QComboBox, QCheckBox)
from PyQt5.QtCore import Qt

# Add the 'Isoharmonics' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tetrahedron_generator import generate_tetrahedron_data, generate_odd_limit_points
from widgets.tetrahedron_widget import TetrahedronWidget
from theory.calculations import generate_ji_tetra_labels

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

        # View Mode Dropdown
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Scatter Plot", "Labels", "Volume Data"])
        self.view_mode_combo.currentTextChanged.connect(self._update_visibility)
        control_layout.addWidget(self.view_mode_combo)

        # Resolution
        self.resolution_widget = QWidget()
        res_layout = QHBoxLayout(self.resolution_widget)
        res_layout.setContentsMargins(0,0,0,0)
        res_label = QLabel("Resolution:")
        self.resolution_input = QSpinBox()
        self.resolution_input.setRange(10, 300)
        self.resolution_input.setValue(60)
        self.resolution_input.setToolTip("Higher values increase detail but require much more computation time.")
        res_layout.addWidget(res_label)
        res_layout.addWidget(self.resolution_input)
        control_layout.addWidget(self.resolution_widget)

        # Limit Mode Dropdown
        self.limit_mode_layout_widget = QWidget()
        limit_mode_layout = QHBoxLayout(self.limit_mode_layout_widget)
        limit_mode_layout.setContentsMargins(0,0,0,0)
        limit_mode_label = QLabel("Limit Mode:")
        self.limit_mode_combo = QComboBox()
        self.limit_mode_combo.addItems(["Odd Limit", "Integer Limit"])
        self.limit_mode_combo.currentTextChanged.connect(self._update_limit_mode)
        limit_mode_layout.addWidget(limit_mode_label)
        limit_mode_layout.addWidget(self.limit_mode_combo)
        control_layout.addWidget(self.limit_mode_layout_widget)
        self.limit_mode = "odd"

        # Odd Limit
        self.odd_limit_layout_widget = QWidget()
        self.odd_limit_layout = QHBoxLayout(self.odd_limit_layout_widget)
        self.odd_limit_layout.setContentsMargins(0,0,0,0)
        odd_limit_label = QLabel("Odd-Limit:")
        self.odd_limit_input = QLineEdit("9")
        self.odd_limit_layout.addWidget(odd_limit_label)
        self.odd_limit_layout.addWidget(self.odd_limit_input)
        control_layout.addWidget(self.odd_limit_layout_widget)

        # Complexity Measures Dropdown
        self.complexity_widget = QWidget()
        complexity_layout = QHBoxLayout(self.complexity_widget)
        complexity_layout.setContentsMargins(0,0,0,0)
        complexity_label = QLabel("Complexity Measures:")
        self.complexity_combo = QComboBox()
        self.complexity_combo.addItems(["Gradus", "Tenney", "Weil", "Wilson", "Off"])
        self.complexity_combo.setCurrentText("Tenney")
        self.complexity_combo.currentTextChanged.connect(self._update_complexity_measure)
        complexity_layout.addWidget(complexity_label)
        complexity_layout.addWidget(self.complexity_combo)
        control_layout.addWidget(self.complexity_widget)
        self.complexity_measure = "Tenney"

        # Size Input
        self.size_widget = QWidget()
        size_layout = QHBoxLayout(self.size_widget)
        size_layout.setContentsMargins(0,0,0,0)
        size_label = QLabel("Size:")
        self.size_input = QLineEdit("1.0")
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.size_input)
        control_layout.addWidget(self.size_widget)

        # Feature Scaling Input
        self.feature_scaling_widget = QWidget()
        feature_scaling_layout = QHBoxLayout(self.feature_scaling_widget)
        feature_scaling_layout.setContentsMargins(0,0,0,0)
        feature_scaling_label = QLabel("Feature Scaling:")
        self.feature_scaling_input = QLineEdit("10")
        feature_scaling_layout.addWidget(feature_scaling_label)
        feature_scaling_layout.addWidget(self.feature_scaling_input)
        control_layout.addWidget(self.feature_scaling_widget)

        # Omission Checkboxes
        self.omissions_widget = QWidget()
        omissions_layout = QHBoxLayout(self.omissions_widget)
        omissions_layout.setContentsMargins(0,0,0,0)
        self.omit_unisons_checkbox = QCheckBox("Omit Unisons")
        self.omit_octaves_checkbox = QCheckBox("Omit Octaves")
        omissions_layout.addWidget(self.omit_unisons_checkbox)
        omissions_layout.addWidget(self.omit_octaves_checkbox)
        control_layout.addWidget(self.omissions_widget)

        self._update_visibility("Scatter Plot")
        self.view_mode_combo.setCurrentIndex(0)

        control_layout.addStretch()

        # Render Button
        self.render_button = QPushButton("Render")
        self.render_button.clicked.connect(self.update_visualization)
        control_layout.addWidget(self.render_button)

        dock_widget = QDockWidget("Controls", self)
        dock_widget.setWidget(control_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_widget)

    def _update_limit_mode(self, text):
        if text == "Odd Limit":
            self.limit_mode = "odd"
            self.odd_limit_input.setToolTip("Odd-Limit must be an odd number >= 3.")
        elif text == "Integer Limit":
            self.limit_mode = "integer"
            self.odd_limit_input.setToolTip("Integer-Limit must be an integer >= 1.")
        # No auto-update

    def _update_complexity_measure(self, text):
        self.complexity_measure = text
        # No auto-update

    def _update_visibility(self, view_mode):
        is_volume = (view_mode == "Volume Data")
        is_scatter = (view_mode == "Scatter Plot")
        is_labels = (view_mode == "Labels")
        is_scatter_or_labels = is_scatter or is_labels

        self.resolution_widget.setVisible(is_volume)
        self.limit_mode_layout_widget.setVisible(is_scatter_or_labels)
        self.odd_limit_layout_widget.setVisible(is_scatter_or_labels)
        self.complexity_widget.setVisible(is_scatter_or_labels)
        self.size_widget.setVisible(is_scatter_or_labels)
        self.feature_scaling_widget.setVisible(is_scatter_or_labels)
        self.omissions_widget.setVisible(is_scatter_or_labels)

        if is_scatter:
            self.feature_scaling_input.setText("10")
        elif is_labels:
            self.feature_scaling_input.setText("7")

    def update_visualization(self):
        try:
            equave_ratio = float(self.equave_input.text())
            if equave_ratio <= 1:
                raise ValueError("Equave must be a positive number > 1.")
            
            view_mode = self.view_mode_combo.currentText()
            show_volume = (view_mode == "Volume Data")
            show_points = (view_mode == "Scatter Plot")
            show_labels = (view_mode == "Labels")
            
            limit_value = 0
            current_limit_mode = self.limit_mode
            universal_scale = 1.0
            feature_scaling = 1.0
            omit_unisons = False
            omit_octaves = False
            
            if show_points or show_labels:
                limit_value = int(self.odd_limit_input.text())
                universal_scale = float(self.size_input.text())
                feature_scaling = float(self.feature_scaling_input.text())
                omit_unisons = self.omit_unisons_checkbox.isChecked()
                omit_octaves = self.omit_octaves_checkbox.isChecked()
                if current_limit_mode == "odd":
                    if limit_value < 3 or limit_value % 2 == 0:
                        raise ValueError("Odd-Limit must be an odd number >= 3.")
                elif current_limit_mode == "integer":
                    if limit_value < 1:
                        raise ValueError("Integer-Limit must be an integer >= 1.")

        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            return

        resolution = self.resolution_input.value()
        
        progress = QProgressDialog("Generating Visualization Data...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()

        volume_data = None
        if show_volume:
            volume_data = generate_tetrahedron_data(equave_ratio, resolution)
            if volume_data is None or volume_data[3] is None:
                QMessageBox.warning(self, "Generation Error", "Could not generate HE data for coordinate system.")
                progress.close()
                return
        
        progress.setValue(50)
        QApplication.processEvents()

        points_data = None
        if show_points:
            points_data = generate_odd_limit_points(
                limit_value, equave_ratio, 
                limit_mode=current_limit_mode, 
                complexity_measure=self.complexity_measure,
                hide_unison_voices=omit_unisons,
                omit_octaves=omit_octaves
            )

        labels_data = None
        if show_labels:
            labels_data = generate_ji_tetra_labels(
                limit_value, equave_ratio, 
                limit_mode=current_limit_mode, 
                complexity_measure=self.complexity_measure,
                hide_unison_voices=omit_unisons,
                omit_octaves=omit_octaves
            )

        self.tetra_widget.update_tetrahedron(
            volume_data=volume_data, 
            points_data=points_data,
            labels_data=labels_data,
            show_volume=show_volume,
            show_points=show_points,
            show_labels=show_labels,
            universal_scale=universal_scale,
            feature_scaling=feature_scaling,
            complexity_measure=self.complexity_measure
        )
        
        progress.setValue(100)
        progress.close()
