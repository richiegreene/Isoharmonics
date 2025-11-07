from PyQt5.QtWidgets import QWidget, QMainWindow, QSplitter, QVBoxLayout, QPushButton, QLabel, QLineEdit, QHBoxLayout, QToolButton, QSpacerItem, QSizePolicy, QFileDialog, QButtonGroup, QInputDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFontDatabase, QFont, QImage, QPainter, QPainterPath, QPolygonF, QBrush, QColor
from fractions import Fraction
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from gui.widgets.isohe_widget import IsoHEWidget
from theory.triangle_generator import generate_triangle_image
from theory.calculations import calculate_edo_step
from theory.notation.engine import calculate_single_note
from theory.sethares import get_dissonance_data_3d_raw, transform_and_interpolate_to_triangle

class SetharesWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, spectrum_data, ref_freq, max_interval, step_size_3d, grid_resolution, z_axis_ramp, cents_spread):
        super().__init__()
        self.spectrum_data = spectrum_data
        self.ref_freq = ref_freq
        self.max_interval = max_interval
        self.step_size_3d = step_size_3d
        self.grid_resolution = grid_resolution
        self.z_axis_ramp = z_axis_ramp
        self.cents_spread = cents_spread

    def run(self):
        r_raw, s_raw, z_raw = get_dissonance_data_3d_raw(
            self.spectrum_data, self.ref_freq, self.max_interval, self.step_size_3d
        )
        x_tri_grid, y_tri_grid, z_tri_interpolated = transform_and_interpolate_to_triangle(
            r_raw, s_raw, z_raw, self.max_interval, self.grid_resolution, self.z_axis_ramp, self.cents_spread
        )
        self.finished.emit((x_tri_grid, y_tri_grid, z_tri_interpolated))

class TriadsWindow(QMainWindow):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.setWindowTitle("Triadic Models")
        self.setGeometry(150, 150, 600, 550)
        self.setStyleSheet("background-color: #23262F;")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setFocusPolicy(Qt.StrongFocus)
        self.sidebar_width = 200

        # Colormaps
        self.colors = ["#23262F", "#1E1861", "#1A0EBE", "#0437f2", "#7895fc", "#A7C6ED", "#D0E1F9", "#F0F4FF", "#FFFFFF"]
        self.custom_cm = LinearSegmentedColormap.from_list("color_gradient", self.colors)

        # Caches
        self.image_banks = {'blank': None, 'harmonic_entropy': None, 'sethares': None}
        self.data_banks = {'harmonic_entropy': None, 'sethares': None}
        
        self.current_bank_index = 0
        self.bank_order = ['blank', 'harmonic_entropy', 'sethares']

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

        # Styles
        self.tool_button_style = """
            QToolButton { background: #343744; border: none; padding: 5px; }
            QToolButton:hover { background: #404552; }
        """
        self.checkable_button_style = """
            QPushButton {
                background-color: #2C2F3B; color: white;
                border: 1px solid #555; padding: 4px; border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #0437f2; color: white;
            }
            QPushButton:disabled {
                background-color: #2C2F3B; color: #555; border: 1px solid #444;
            }
        """
        self.non_checkable_button_style = """
            QPushButton {
                background-color: #2C2F3B; color: white;
                border: 1px solid #555; padding: 4px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #404552; }
        """

        self.collapse_button = QToolButton()
        self.collapse_button.setStyleSheet(self.tool_button_style)
        self.collapse_button.setArrowType(Qt.RightArrow)
        self.collapse_button.clicked.connect(self.toggle_sidebar)
        self.sidebar_layout.addWidget(self.collapse_button)
        
        # Harmonic Entropy
        self.harmonic_entropy_button = QPushButton("Harmonic Entropy")
        self.harmonic_entropy_button.setStyleSheet(self.non_checkable_button_style)
        self.harmonic_entropy_button.clicked.connect(self.generate_triangle_image)
        self.sidebar_layout.addWidget(self.harmonic_entropy_button)

        # Sethares Model
        self.sethares_model_button = QPushButton("Sethares Model")
        self.sethares_model_button.setStyleSheet(self.non_checkable_button_style)
        self.sethares_model_button.clicked.connect(self.generate_sethares_model)
        self.sidebar_layout.addWidget(self.sethares_model_button)

        # Topographic Lines Button
        self.topo_button = QPushButton("Lines")
        self.topo_button.setStyleSheet(self.checkable_button_style)
        self.topo_button.setCheckable(True)
        self.topo_button.toggled.connect(self.display_current_image)
        self.sidebar_layout.addWidget(self.topo_button)

        # EDO Button
        self.edo_button = QPushButton("EDO")
        self.edo_button.setStyleSheet(self.checkable_button_style)
        self.edo_button.setCheckable(True)
        self.edo_button.toggled.connect(self.toggle_edo_dots)
        self.sidebar_layout.addWidget(self.edo_button)

        # Labels Button
        self.labels_button = QPushButton("Labels")
        self.labels_button.setStyleSheet(self.checkable_button_style)
        self.labels_button.setCheckable(True)
        self.labels_button.toggled.connect(self.toggle_edo_labels)
        self.labels_button.setEnabled(False)
        self.sidebar_layout.addWidget(self.labels_button)
        
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
            button.setStyleSheet(self.checkable_button_style)
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

        # Last Generated
        last_generated_layout = QVBoxLayout()
        last_generated_label = QLabel("Last Generated")
        last_generated_label.setStyleSheet("color: white;")
        last_generated_layout.addWidget(last_generated_label)

        button_layout = QHBoxLayout()
        self.he_button = QPushButton("HE")
        self.sm_button = QPushButton("SM")
        self.blank_button = QPushButton("Blank")

        for button in [self.he_button, self.sm_button, self.blank_button]:
            button.setStyleSheet(self.non_checkable_button_style)
            button_layout.addWidget(button)

        self.he_button.clicked.connect(lambda: self.set_current_model('harmonic_entropy'))
        self.sm_button.clicked.connect(lambda: self.set_current_model('sethares'))
        self.blank_button.clicked.connect(lambda: self.set_current_model('blank'))

        self.model_button_group = QButtonGroup()
        self.model_button_group.setExclusive(True)
        self.model_button_group.addButton(self.he_button)
        self.model_button_group.addButton(self.sm_button)
        self.model_button_group.addButton(self.blank_button)

        last_generated_layout.addLayout(button_layout)
        self.sidebar_layout.addLayout(last_generated_layout)

        self.sidebar_layout.addStretch()

        # Save buttons
        save_button_layout = QHBoxLayout()
        self.save_png_button = QPushButton(".png")
        self.save_png_button.setStyleSheet(self.non_checkable_button_style)
        self.save_png_button.clicked.connect(self.save_triangle_image)
        save_button_layout.addWidget(self.save_png_button)

        self.save_svg_button = QPushButton(".svg")
        self.save_svg_button.setStyleSheet(self.non_checkable_button_style)
        self.save_svg_button.clicked.connect(self.save_svg)
        save_button_layout.addWidget(self.save_svg_button)

        self.save_obj_button = QPushButton(".obj")
        self.save_obj_button.setStyleSheet(self.non_checkable_button_style)
        self.save_obj_button.clicked.connect(self.save_obj)
        save_button_layout.addWidget(self.save_obj_button)
        self.sidebar_layout.addLayout(save_button_layout)

        self.loading_label = QLabel()
        self.loading_label.setStyleSheet("color: grey;")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.sidebar_layout.addWidget(self.loading_label)
        self.loading_label.hide()

        self.expand_button = QToolButton(self)
        self.expand_button.setStyleSheet(self.tool_button_style)
        self.expand_button.setArrowType(Qt.LeftArrow)
        self.expand_button.setFixedSize(30, 30)
        self.expand_button.clicked.connect(self.toggle_sidebar)
        self.expand_button.show()

        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.isohe_widget)
        self.splitter.setSizes([0, self.width()])
        
        self.current_pivot = "1"
        self.pivot_buttons[self.current_pivot].setChecked(True)

        self.update_equave()
        self.collapse_sidebar()

    def save_svg(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save SVG", "", "SVG Files (*.svg)")
        if file_path:
            topo_data = None
            model_name = self.bank_order[self.current_bank_index]
            if self.topo_button.isChecked():
                if model_name in self.data_banks:
                    topo_data = self.data_banks[model_name]
            self.isohe_widget.save_svg(file_path, topo_data, self.custom_cm, model_name)

    def generate_topographic_image(self, model_name):
        if model_name not in self.data_banks or self.data_banks[model_name] is None:
            return None

        X, Y, Z = self.data_banks[model_name]
        
        dpi = 150
        levels = 15
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        if np.isnan(z_min) or np.isnan(z_max): return None
        contour_levels = np.linspace(z_min, z_max, levels)

        if model_name == 'harmonic_entropy':
            grid_height, grid_width = Z.shape
            fig, ax = plt.subplots(figsize=(grid_width / dpi, grid_height / dpi), dpi=dpi)
            ax.contour(X, Y, Z, levels=contour_levels, cmap=self.custom_cm, linewidths=2, origin='lower')
            ax.set_aspect('auto')
        
        elif model_name == 'sethares':
            aspect_ratio = np.sqrt(3) / 2
            fig, ax = plt.subplots(figsize=(8, 8 * aspect_ratio), dpi=dpi)
            ax.contour(X, Y, Z, levels=contour_levels, cmap=self.custom_cm, linewidths=2)
            ax.set_aspect('equal')

        else:
            return None

        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        q_image = QImage.fromData(buf.read())
        plt.close(fig)
        
        return q_image

    def save_obj(self):
        current_model_name = self.bank_order[self.current_bank_index]
        if current_model_name not in self.data_banks or self.data_banks[current_model_name] is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save OBJ", "", "OBJ Files (*.obj)")
        if not file_path:
            return

        X, Y, Z = self.data_banks[current_model_name]
        
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        Z_normalized = 100 * (Z - z_min) / (z_max - z_min) if z_max - z_min > 0 else np.zeros_like(Z)

        rows, cols = X.shape
        vertices = []
        faces = []
        
        vertex_map = {}
        vertex_idx = 1

        for r in range(rows):
            for c in range(cols):
                if not np.isnan(Z_normalized[r, c]):
                    vertices.append(f"v {X[r, c]} {Y[r, c]} {Z_normalized[r, c]}\n")
                    vertex_map[(r, c)] = vertex_idx
                    vertex_idx += 1

        for r in range(rows - 1):
            for c in range(cols - 1):
                v1_idx, v2_idx, v3_idx, v4_idx = (vertex_map.get(p) for p in [(r, c), (r + 1, c), (r + 1, c + 1), (r, c + 1)])
                if all([v1_idx, v2_idx, v3_idx, v4_idx]):
                    faces.append(f"f {v1_idx} {v2_idx} {v3_idx} {v4_idx}\n")
                elif all([v1_idx, v2_idx, v3_idx]):
                    faces.append(f"f {v1_idx} {v2_idx} {v3_idx}\n")
                elif all([v1_idx, v3_idx, v4_idx]):
                    faces.append(f"f {v1_idx} {v3_idx} {v4_idx}\n")

        with open(file_path, 'w') as f:
            f.writelines(vertices)
            f.writelines(faces)

    def mousePressEvent(self, event):
        focused_widget = self.focusWidget()
        if isinstance(focused_widget, QLineEdit):
            focused_widget.clearFocus()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_1: self.set_pivot("1")
        elif key == Qt.Key_2: self.set_pivot("2")
        elif key == Qt.Key_3: self.set_pivot("3")
        else: super().keyPressEvent(event)

    def set_pivot(self, pivot_name):
        if self.current_pivot != pivot_name:
            self.pivot_buttons[self.current_pivot].setChecked(False)
            self.current_pivot = pivot_name
            self.pivot_buttons[pivot_name].setChecked(True)
            self.isohe_widget.set_pivot_voice({"3": "upper", "2": "middle", "1": "lower"}[pivot_name])

    def toggle_edo_dots(self, checked):
        self.isohe_widget.set_show_edo_dots(checked)
        self.labels_button.setEnabled(checked)
        if not checked: self.labels_button.setChecked(False)

    def toggle_edo_labels(self, checked):
        self.isohe_widget.set_show_edo_labels(checked)

    def toggle_sidebar(self):
        self.expand_sidebar() if self.sidebar.width() == 0 else self.collapse_sidebar()

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
            row1_idx, row2_idx = int(self.equave_start.text()), int(self.equave_end.text())
            table = self.main_app.table
            if not (0 <= row1_idx < table.rowCount() and 0 <= row2_idx < table.rowCount()): return
            item1, item2 = table.item(row1_idx, 1), table.item(row2_idx, 1)
            if item1 and item2:
                ratio1, ratio2 = Fraction(item1.text()), Fraction(item2.text())
                if ratio1 != 0: self.isohe_widget.set_equave(ratio2 / ratio1)
        except (ValueError, ZeroDivisionError): pass

    def closeEvent(self, event):
        self.isohe_widget.stop_sound()
        super().closeEvent(event)

    def set_current_model(self, bank_name):
        self.current_bank_index = self.bank_order.index(bank_name)
        button = self.model_button_group.button(self.current_bank_index)
        if button: button.setChecked(True)
        self.display_current_image()

    def display_current_image(self):
        bank_name = self.bank_order[self.current_bank_index]
        
        if self.topo_button.isChecked():
            if bank_name in self.data_banks:
                topo_data = self.data_banks[bank_name]
                self.isohe_widget.set_topo_data(topo_data, self.custom_cm, bank_name)
            else:
                self.isohe_widget.clear_topo_data()
                self.isohe_widget.set_triangle_image(None)
        else:
            self.isohe_widget.clear_topo_data()
            image = self.image_banks.get(bank_name)
            self.isohe_widget.set_triangle_image(image)

    def save_triangle_image(self):
        current_model_name = self.bank_order[self.current_bank_index]
        image_to_save = None

        if self.topo_button.isChecked():
            # If topo is checked, we need to generate the image with vector data for saving
            if current_model_name in self.data_banks:
                image_to_save = self.generate_topographic_image(current_model_name)
        else:
            image_to_save = self.isohe_widget.triangle_image

        if image_to_save is None: return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
        if not file_path: return

        width, height = image_to_save.width(), image_to_save.height()
        cropped_image = QImage(width, height, QImage.Format_ARGB32)
        cropped_image.fill(Qt.transparent)

        path = QPainterPath()
        path.moveTo(width / 2, 0)
        path.lineTo(0, height)
        path.lineTo(width, height)
        path.closeSubpath()

        painter = QPainter(cropped_image)
        painter.setClipPath(path)
        painter.drawImage(0, 0, image_to_save)
        painter.end()
        cropped_image.save(file_path)

    def generate_triangle_image(self):
        if hasattr(self.isohe_widget, 'equave'):
            self.loading_label.setText("Loading Harmonic Entropy...")
            self.loading_label.show()
            image, x, y, z = generate_triangle_image(self.isohe_widget.equave, self.isohe_widget.width(), self.isohe_widget.height())
            
            model_name = 'harmonic_entropy'
            self.image_banks[model_name] = image
            self.data_banks[model_name] = (x, y, z)
            
            self.set_current_model(model_name)
            self.loading_label.hide()

    def generate_sethares_model(self):
        try:
            self.loading_label.setText("Loading...")
            self.loading_label.show()
            ref_freq = float(Fraction(self.main_app.isoharmonic_entry.text())) * 261.6256
            equave_ratio = self.isohe_widget.equave
            max_interval = float(equave_ratio)
            roll_off_rate = self.main_app.roll_off_rate

            current_timbre = self.main_app.visualizer.current_timbre
            if current_timbre in [self.main_app.ji_timbre, self.main_app.edo_timbre]:
                partials = current_timbre['ratios']
            else:
                partials = [1.0]

            amplitudes = [(1.0 / (fr**roll_off_rate) if roll_off_rate > 0 else fr**abs(roll_off_rate)) if fr != 0 else 0.0 for fr in partials]
            if roll_off_rate == 0: amplitudes = [1.0 for _ in partials]

            spectrum_data = {'freq': partials, 'amp': amplitudes}
            
            self.worker = SetharesWorker(spectrum_data, ref_freq, max_interval, 0.01, 400, 2.0, 0)
            self.worker.finished.connect(self.on_sethares_finished)
            self.worker.start()
        except Exception as e:
            print(f"Error starting Sethares model generation: {e}")
            self.loading_label.hide()

    def on_sethares_finished(self, result):
        try:
            x_grid, y_grid, z_interpolated = result
            model_name = 'sethares'
            self.data_banks[model_name] = result

            aspect_ratio = np.sqrt(3) / 2
            fig, ax = plt.subplots(figsize=(8, 8 * aspect_ratio), dpi=150)
            ax.imshow(z_interpolated, cmap=self.custom_cm, origin='lower', extent=[0, 1200, 0, 1200 * aspect_ratio], aspect='equal')
            ax.axis('off')
            fig.tight_layout(pad=0)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', transparent=True)
            buf.seek(0)
            q_image = QImage.fromData(buf.read())
            plt.close(fig)

            self.image_banks[model_name] = q_image
            
            self.set_current_model(model_name)
        except Exception as e:
            print(f"Error displaying Sethares model: {e}")
        finally:
            self.loading_label.hide()