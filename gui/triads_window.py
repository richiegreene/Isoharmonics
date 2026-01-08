from PyQt5.QtWidgets import QWidget, QMainWindow, QSplitter, QVBoxLayout, QPushButton, QLabel, QLineEdit, QHBoxLayout, QToolButton, QSpacerItem, QSizePolicy, QFileDialog, QButtonGroup, QInputDialog, QComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFontDatabase, QFont, QImage, QPainter, QPainterPath, QPolygonF, QBrush, QColor, QVector3D
from fractions import Fraction
import numpy as np
import io
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from gui.widgets.isohe_widget import IsoHEWidget
from theory.triangle_generator import generate_triangle_image
from theory.calculations import calculate_edo_step
from theory.notation.engine import calculate_single_note
from theory.sethares import get_dissonance_data_3d_raw, transform_and_interpolate_to_triangle
from utils.formatters import to_subscript

# Helper function for point-in-triangle test
def is_point_in_triangle(px, py, v1x, v1y, v2x, v2y, v3x, v3y):
    # Barycentric coordinates method
    # Using a small epsilon for boundary checks to be inclusive
    epsilon = 1e-9

    s = v1y * v3x - v1x * v3y + (v3y - v1y) * px + (v1x - v3x) * py
    t = v1x * v2y - v1y * v2x + (v1y - v2y) * px + (v2x - v1x) * py

    # Denominator for barycentric coordinates
    A = -v2y * v3x + v1y * (v3x - v2x) + v1x * (v2y - v3y) + v2x * v3y
    
    # Handle degenerate triangle or division by zero
    if abs(A) < epsilon:
        return False
    
    A_inv = 1 / A

    s = s * A_inv
    t = t * A_inv
    
    # Check if point is inside or on the edges (inclusive)
    return (s >= -epsilon) and (t >= -epsilon) and (1 - s - t >= -epsilon)

# Optional 3D imports
try:
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import MeshData, GLMeshItem, GLViewWidget, GLScatterPlotItem, GLTextItem
except Exception:
    gl = None

try:
    from OpenGL.GL import glGetDoublev, GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, glGetIntegerv, GL_VIEWPORT
    from OpenGL.GLU import gluProject
except Exception:
    # If PyOpenGL is not installed, picking will be disabled gracefully
    glGetDoublev = None
    gluProject = None

try:
    # additional GL material and lighting functions for shininess/specular control
    from OpenGL.GL import (
        glMaterialfv, GL_FRONT_AND_BACK, GL_SPECULAR, GL_SHININESS,
        glEnable, glDisable, GL_COLOR_MATERIAL, glColorMaterial, GL_AMBIENT_AND_DIFFUSE,
        GL_LIGHTING, GL_LIGHT0, glLightfv, GL_POSITION, GL_DIFFUSE, GL_AMBIENT, GL_SPECULAR as _GL_SPECULAR,
        glPushAttrib, glPopAttrib, GL_LIGHTING_BIT, glShadeModel, GL_SMOOTH, glNormal3f, GL_NORMALIZE
    )
except Exception:
    glMaterialfv = None
    GL_FRONT_AND_BACK = None
    GL_SPECULAR = None
    GL_SHININESS = None
    glEnable = None
    glDisable = None
    GL_COLOR_MATERIAL = None
    glColorMaterial = None
    GL_AMBIENT_AND_DIFFUSE = None
    GL_LIGHTING = None
    GL_LIGHT0 = None
    glLightfv = None
    GL_POSITION = None
    GL_DIFFUSE = None
    GL_AMBIENT = None
    _GL_SPECULAR = None
    glPushAttrib = None
    glPopAttrib = None
    GL_LIGHTING_BIT = None
    glShadeModel = None
    GL_SMOOTH = None
    glNormal3f = None
    GL_NORMALIZE = None

# Subclass GLMeshItem to set OpenGL material properties for increased shininess where available
_ShinyMeshClass = None
if 'GLMeshItem' in globals() and glMaterialfv is not None:
    try:
        class ShinyMesh(GLMeshItem):
            def __init__(self, *args, specular=(1.0, 1.0, 1.0), shininess=80.0, **kwargs):
                super().__init__(*args, **kwargs)
                self._specular = tuple(specular)
                # Clamp shininess to the valid OpenGL range [0, 128]
                self._shininess = max(0.0, min(float(shininess), 128.0))

            def setup_lighting(self):
                if glEnable is not None and GL_LIGHTING is not None:
                    glEnable(GL_LIGHTING)
                if glEnable is not None and GL_LIGHT0 is not None:
                    glEnable(GL_LIGHT0)
                if glEnable is not None and GL_NORMALIZE is not None:
                    glEnable(GL_NORMALIZE)

                if glLightfv is not None and GL_LIGHT0 is not None and GL_POSITION is not None:
                    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 5.0, 5.0, 1.0))
                if glLightfv is not None and GL_LIGHT0 is not None and GL_DIFFUSE is not None:
                    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
                if glLightfv is not None and GL_LIGHT0 is not None and GL_SPECULAR is not None:
                    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))

                if glEnable is not None and GL_COLOR_MATERIAL is not None:
                    glEnable(GL_COLOR_MATERIAL)
                    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

                if glMaterialfv is not None and GL_FRONT_AND_BACK is not None and GL_SPECULAR is not None:
                    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (*self._specular, 1.0))
                if glMaterialfv is not None and GL_FRONT_AND_BACK is not None and GL_SHININESS is not None:
                    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, self._shininess)

            def draw(self):
                self.setup_lighting()
                res = super().draw()
                return res

            def paint(self):
                self.setup_lighting()
                res = super().paint()
                return res

        _ShinyMeshClass = ShinyMesh
    except Exception:
        _ShinyMeshClass = None

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

        # Colormaps: use the standard topo gradient used elsewhere so 2D and 3D
        # share the same color mapping. This matches the original UI gradient.
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

        # spacer to separate generation buttons from view-mode buttons
        spacer_before_3d = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.sidebar_layout.addItem(spacer_before_3d)

        # 3D View Button
        self.view3d_button = QPushButton("3D")
        self.view3d_button.setStyleSheet(self.checkable_button_style)
        self.view3d_button.setCheckable(True)
        self.view3d_button.toggled.connect(self.toggle_3d_view)
        # disable if pyqtgraph.opengl or PyOpenGL not available
        self.view3d_button.setEnabled(gl is not None and glGetDoublev is not None and gluProject is not None)
        if not self.view3d_button.isEnabled():
            self.view3d_button.setToolTip('3D view requires pyqtgraph.opengl and PyOpenGL')
        self.sidebar_layout.addWidget(self.view3d_button)

        # Topographic Lines Button
        self.topo_button = QPushButton("Lines")
        self.topo_button.setStyleSheet(self.checkable_button_style)
        self.topo_button.setCheckable(True)
        self.topo_button.toggled.connect(self.display_current_image)
        # ensure mutual exclusivity between Lines and 3D
        self.topo_button.toggled.connect(lambda checked: (self.view3d_button.setChecked(False) if checked else None))
        self.view3d_button.toggled.connect(lambda checked: (self.topo_button.setChecked(False) if checked else None))
        self.sidebar_layout.addWidget(self.topo_button)

        # spacer after view-mode buttons
        spacer_after_3d = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.sidebar_layout.addItem(spacer_after_3d)

        self.dots_button = QPushButton("Dots")
        self.dots_button.setStyleSheet(self.checkable_button_style)
        self.dots_button.setCheckable(True)
        self.dots_button.toggled.connect(self.toggle_dots)
        
        # Labels Button
        self.labels_button = QPushButton("Labels")
        self.labels_button.setStyleSheet(self.checkable_button_style)
        self.labels_button.setCheckable(True)
        self.labels_button.toggled.connect(self.toggle_dots_labels)

        self.mode_button = QPushButton("JI")
        self.mode_button.setStyleSheet(self.checkable_button_style)
        self.mode_button.setCheckable(True)
        self.mode_button.setChecked(False) # JI is default
        self.mode_button.toggled.connect(self.toggle_dots_mode)

        dots_layout = QHBoxLayout()
        dots_layout.addWidget(self.dots_button)
        dots_layout.addWidget(self.labels_button)
        dots_layout.addWidget(self.mode_button)
        self.sidebar_layout.addLayout(dots_layout)

        # Limit Type Dropdown
        self.limit_type_dropdown = QComboBox()
        self.limit_type_dropdown.addItems(["Odd-Limit", "Integer-Limit", "Prime-Limit"])
        self.limit_type_dropdown.setCurrentText("Odd-Limit") # Default to Odd-Limit
        self.limit_type_dropdown.setStyleSheet("""
            QComboBox {
                background-color: #2C2F3B; color: white;
                border: 1px solid #444; border-radius: 4px; padding: 2px; padding-left: 6px;
            }
            QComboBox::drop-down {
                border: 0px; /* No border for the arrow part */
            }
            QComboBox::down-arrow {
                image: url(./path/to/your/down_arrow.png); /* Replace with actual path to an arrow icon */
                width: 10px;
                height: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2C2F3B;
                color: white;
                selection-background-color: #0437f2;
            }
        """)
        self.sidebar_layout.addWidget(self.limit_type_dropdown)
        self.limit_type_dropdown.currentIndexChanged.connect(self.update_limit_mode)

        # Limit Value Entry
        self.limit_value_entry = QLineEdit("15") # Renamed from odd_limit_entry
        self.limit_value_entry.setStyleSheet("background-color: #2C2F3B; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px; padding-left: 6px;")
        self.limit_value_entry.textChanged.connect(self.update_limit_value) # New method name
        self.sidebar_layout.addWidget(self.limit_value_entry)

        # Prime Limit Inputs
        self.prime_limit_widget = QWidget()
        prime_limit_layout = QVBoxLayout(self.prime_limit_widget)
        prime_limit_layout.setContentsMargins(0, 0, 0, 0)
        prime_limit_layout.setSpacing(4)
        self.prime_limit_label = QLabel("Prime limit")
        self.prime_limit_label.setStyleSheet("color: white;")
        prime_limit_layout.addWidget(self.prime_limit_label)
        self.prime_limit_entry = QLineEdit("7")
        self.prime_limit_entry.setStyleSheet("background-color: #2C2F3B; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px; padding-left: 6px;")
        self.prime_limit_entry.textChanged.connect(self.update_prime_limit)
        prime_limit_layout.addWidget(self.prime_limit_entry)
        self.max_exponent_label = QLabel("Maximum exponent")
        self.max_exponent_label.setStyleSheet("color: white;")
        prime_limit_layout.addWidget(self.max_exponent_label)
        self.max_exponent_entry = QLineEdit("4")
        self.max_exponent_entry.setStyleSheet("background-color: #2C2F3B; color: white; border: 1px solid #444; border-radius: 4px; padding: 2px; padding-left: 6px;")
        self.max_exponent_entry.textChanged.connect(self.update_prime_limit)
        prime_limit_layout.addWidget(self.max_exponent_entry)
        self.sidebar_layout.addWidget(self.prime_limit_widget)
        self.prime_limit_widget.hide()
        
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
        self.isohe_widget.set_dots_mode('JI')
        self.isohe_widget.set_limit_mode("odd") # Set initial mode
        self.isohe_widget.set_limit_value(15) # Set initial value

        # Create 3D view (hidden by default). Use a container so we can swap between 2D and 3D
        self.content_container = QWidget()
        content_layout = QVBoxLayout(self.content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self.isohe_widget)

        self.view3d_widget = None
        if gl is not None:
            try:
                self.view3d_widget = GLViewWidget()
                self.view3d_widget.setBackgroundColor((35, 38, 47))
                self.view3d_widget.hide()
                content_layout.addWidget(self.view3d_widget)
                # store current mesh item so we can remove/replace
                self._3d_mesh_item = None
                self._3d_picking_vertex_list = None
                self._3d_index_map = None
                # Intercept mouse handlers. Require Shift to rotate/drag the 3D view.
                # Click without Shift will perform a pick & play action.
                # Keep originals so we can forward events when Shift is held.
                self._view3d_original_mouse_press = self.view3d_widget.mousePressEvent
                self._view3d_original_mouse_move = getattr(self.view3d_widget, 'mouseMoveEvent', None)
                self._view3d_original_mouse_release = getattr(self.view3d_widget, 'mouseReleaseEvent', None)
                self._3d_rotating = False

                def _view3d_mouse_press(ev):
                    try:
                        if ev.modifiers() & Qt.ShiftModifier:
                            self._3d_rotating = True
                            if self._view3d_original_mouse_press:
                                self._view3d_original_mouse_press(ev)
                        else:
                            self._on_view3d_mouse_press(ev)
                    except Exception:
                        return

                def _view3d_mouse_move(ev):
                    try:
                        if self._3d_rotating:
                            if self._view3d_original_mouse_move:
                                self._view3d_original_mouse_move(ev)
                        else:
                            self._on_view3d_mouse_move(ev)
                    except Exception:
                        return

                def _view3d_mouse_release(ev):
                    try:
                        if self._3d_rotating:
                            if self._view3d_original_mouse_release:
                                self._view3d_original_mouse_release(ev)
                        else:
                            self._on_view3d_mouse_release(ev)
                        self._3d_rotating = False
                    except Exception:
                        self._3d_rotating = False

                self.view3d_widget.mousePressEvent = _view3d_mouse_press
                self.view3d_widget.mouseMoveEvent = _view3d_mouse_move
                self.view3d_widget.mouseReleaseEvent = _view3d_mouse_release
            except Exception:
                self.view3d_widget = None

        
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
        self.splitter.addWidget(self.content_container)
        self.splitter.setSizes([0, self.width()])
        
        self.current_pivot = "1"
        self.pivot_buttons[self.current_pivot].setChecked(True)

        self.update_equave()
        self.collapse_sidebar()

        self._3d_transform_params = {}
        self._3d_edo_dots_item = None
        self._3d_edo_labels_items = [] # Initialize for 3D EDO labels
        self._debug_marker = None

    def toggle_dots(self, checked):
        self.isohe_widget.set_show_dots(checked)
        # self.labels_button.setEnabled(checked) # This line was removed in the previous step, but double check
        # if not checked: self.labels_button.setChecked(False) # REMOVED THIS LINE

        # Handle 3D EDO dots
        if self.view3d_button.isChecked() and self.view3d_widget is not None:
            if checked:
                self._add_edo_dots_to_3d_view()
            else:
                if self._3d_edo_dots_item is not None:
                    try:
                        self.view3d_widget.removeItem(self._3d_edo_dots_item)
                    except Exception:
                        pass
                    self._3d_edo_dots_item = None

    def toggle_dots_labels(self, checked):
        self.isohe_widget.set_show_labels(checked)
        
        # Handle 3D EDO labels
        if self.view3d_button.isChecked() and self.view3d_widget is not None:
            if checked:
                # If labels are checked, and dots are visible, re-add dots to trigger label generation
                if self.dots_button.isChecked():
                    self._add_edo_dots_to_3d_view()
            else:
                # If labels are unchecked, remove all existing labels
                for item in self._3d_edo_labels_items:
                    try:
                        self.view3d_widget.removeItem(item)
                    except Exception:
                        pass
                self._3d_edo_labels_items = []

    def toggle_dots_mode(self, checked):
        if checked:
            self.mode_button.setText("EDO")
            # self.odd_limit_label.hide() # REMOVED
            # self.odd_limit_entry.hide() # REMOVED
            self.isohe_widget.set_dots_mode('EDO')
        else:
            self.mode_button.setText("JI")
            # self.odd_limit_label.show() # REMOVED
            # self.odd_limit_entry.show() # REMOVED
            self.isohe_widget.set_dots_mode('JI')
        self.isohe_widget.update()

    def update_odd_limit(self, text):
        try:
            limit = int(text)
            if limit > 0:
                self.isohe_widget.set_odd_limit(limit)
        except ValueError:
            pass

    def save_svg(self):
        if self.view3d_widget and self.view3d_widget.isVisible():
            file_path, _ = QFileDialog.getSaveFileName(self, "Save 3D View as SVG", "", "SVG Files (*.svg)")
            if not file_path:
                return

            image = QImage(self.view3d_widget.size(), QImage.Format_ARGB32)
            image.fill(Qt.transparent)
            painter = QPainter(image)
            self.view3d_widget.render(painter)
            painter.end()

            import base64
            from PyQt5.QtCore import QBuffer, QIODevice
            
            buffer = QBuffer()
            buffer.open(QIODevice.WriteOnly)
            image.save(buffer, "PNG")
            base64_data = base64.b64encode(buffer.data().data()).decode('utf-8')
            
            width = image.width()
            height = image.height()

            svg_content = f"""<svg width=\"{width}\" height=\"{height}\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">
  <image x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" xlink:href=\"data:image/png;base64,{base64_data}\" />
</svg>
"""
            with open(file_path, 'w') as f:
                f.write(svg_content)
            return

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
        levels = 30
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        if np.isnan(z_min) or np.isnan(z_max): return None
        contour_levels = np.linspace(z_min, z_max, levels)

        if model_name == 'harmonic_entropy':
            grid_height, grid_width = Z.shape
            fig, ax = plt.subplots(figsize=(grid_width / dpi, grid_height / dpi), dpi=dpi)
            ax.contour(X, Y, Z, levels=contour_levels, cmap=self.custom_cm, linewidths=4, origin='lower')
            ax.set_aspect('auto')
        
        elif model_name == 'sethares':
            aspect_ratio = np.sqrt(3) / 2
            fig, ax = plt.subplots(figsize=(8, 8 * aspect_ratio), dpi=dpi)
            ax.contour(X, Y, Z, levels=contour_levels, cmap=self.custom_cm, linewidths=4)
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

    def update_limit_mode(self, index):
        mode = self.limit_type_dropdown.itemText(index).replace("-Limit", "").lower()
        self.isohe_widget.set_limit_mode(mode)

        if mode == "prime":
            self.limit_value_entry.hide()
            self.prime_limit_widget.show()
            self.update_prime_limit()
        else:
            self.limit_value_entry.show()
            self.prime_limit_widget.hide()
            self.update_limit_value(self.limit_value_entry.text())

        self.isohe_widget.update() # Force redraw

    def update_limit_value(self, text):
        try:
            limit = int(text)
            if limit > 0:
                self.isohe_widget.set_limit_value(limit)
        except ValueError:
            pass

    def update_prime_limit(self):
        try:
            prime = int(self.prime_limit_entry.text())
            exponent = int(self.max_exponent_entry.text())
            if prime > 0 and exponent > 0:
                self.isohe_widget.set_prime_limit(prime, exponent)
        except ValueError:
            pass

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
        # If 3D view is active, show the GL view; if topo (Lines) is active, show topo contours
        if hasattr(self, 'view3d_button') and self.view3d_button.isChecked() and self.view3d_widget is not None:
            # hide 2D widget and show 3D
            self.isohe_widget.hide()
            self.view3d_widget.show()
            # build mesh from data if available
            if bank_name in self.data_banks and self.data_banks[bank_name] is not None:
                self._build_3d_mesh(bank_name)
            else:
                # no data - clear any existing mesh
                if hasattr(self, '_3d_mesh_item') and self._3d_mesh_item is not None:
                    try:
                        self.view3d_widget.removeItem(self._3d_mesh_item)
                    except Exception:
                        pass
                    self._3d_mesh_item = None
                # Also clear EDO dots if no mesh
                if self._3d_edo_dots_item is not None:
                    try:
                        self.view3d_widget.removeItem(self._3d_edo_dots_item)
                    except Exception:
                        pass
                    self._3d_edo_dots_item = None
                # Also clear EDO labels if no mesh
                for item in self._3d_edo_labels_items:
                    try:
                        self.view3d_widget.removeItem(item)
                    except Exception:
                        pass
                self._3d_edo_labels_items = []
        else:
            # ensure 3D view hidden
            if self.view3d_widget is not None:
                self.view3d_widget.hide()
                # Clear EDO dots when switching to 2D view
                if self._3d_edo_dots_item is not None:
                    try:
                        self.view3d_widget.removeItem(self._3d_edo_dots_item)
                    except Exception:
                        pass
                    self._3d_edo_dots_item = None
                # Clear EDO labels when switching to 2D view
                for item in self._3d_edo_labels_items:
                    try:
                        self.view3d_widget.removeItem(item)
                    except Exception:
                        pass
                self._3d_edo_labels_items = []
            self.isohe_widget.show()

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

    def toggle_3d_view(self, checked):
        # called when the 3D button toggles
        if checked:
            # deselect lines/topo
            if hasattr(self, 'topo_button'):
                self.topo_button.setChecked(False)
        # refresh display
        self.display_current_image()

    def _update_sound_from_cents(self, c1, c2):
        # This is a simplified version of isohe_widget.update_ratios_and_sound
        # It takes cents values directly
        try:
            p_pitch = self.isohe_widget.pivot_pitch
            pivot_voice = self.isohe_widget.pivot_voice

            if pivot_voice == "upper":
                pitch1, pitch2, pitch3 = p_pitch - c1 - c2, p_pitch - c2, p_pitch
            elif pivot_voice == "middle":
                pitch1, pitch2, pitch3 = p_pitch - c1, p_pitch, p_pitch + c2
            else: # lower
                pitch1, pitch2, pitch3 = p_pitch, p_pitch + c1, p_pitch + c1 + c2

            def cents_to_ratio(c): return 2 ** (c / 1200)
            new_ratios = [cents_to_ratio(p) for p in [pitch1, pitch2, pitch3]]
            
            # Update labels
            self.cent_labels["3"].setText(f"{int(round(pitch3))}")
            self.cent_labels["2"].setText(f"{int(round(pitch2))}")
            self.cent_labels["1"].setText(f"{int(round(pitch1))}")

            edo = int(self.main_app.edo_entry.text())
            for i, pitch in enumerate([pitch1, pitch2, pitch3]):
                step_str, error = calculate_edo_step(pitch, edo)
                step = int(step_str.replace("-", "-"))
                note_name = calculate_single_note(step, edo)
                octave = 4 + (pitch // 1200)
                note_name_with_octave = note_name + to_subscript(int(octave))
                error_str = f"{round(-error):+}".replace("-", "-")
                if error_str in ["+0", "-0"]: error_str = ""
                self.note_labels[str(i + 1)].setText(f"{note_name_with_octave} {error_str}")

            self.isohe_widget.current_ratios = new_ratios
            self.isohe_widget.update_ratios_label()

            # Play sound (copied from isohe_widget)
            if not all(np.isfinite(new_ratios)) or any(r <= 0 for r in new_ratios): return
            if self.isohe_widget.last_played_ratios is None or any(abs(a - b) > 1e-6 for a, b in zip(new_ratios, self.isohe_widget.last_played_ratios)):
                # Use the background playback method
                if hasattr(self.isohe_widget, 'play_current_ratios_background'):
                    self.isohe_widget.play_current_ratios_background()
                else:
                    self.isohe_widget.update_sound() # Fallback
                self.isohe_widget.last_played_ratios = list(new_ratios)
        except Exception as e:
            print(f"Error in _update_sound_from_cents: {e}")

    def _pick_and_play_3d(self, pos):
        # Map click on GLViewWidget to nearest mesh vertex and play via isohe_widget
        try:
            if pos is None: return
            if not hasattr(self, '_3d_picking_vertex_list') or self._3d_picking_vertex_list is None:
                return
            if gluProject is None:
                return

            # Use pyqtgraph's methods to get matrices and viewport
            viewport = self.view3d_widget.getViewport()
            proj = self.view3d_widget.projectionMatrix()
            model = self.view3d_widget.viewMatrix()

            proj_np = np.array(proj.data(), dtype=np.float64).reshape(4, 4)
            model_np = np.array(model.data(), dtype=np.float64).reshape(4, 4)

            best_idx = None
            best_dist2 = float('inf')
            mx, my = pos.x(), pos.y()

            # Optimization: check only a subset of vertices to avoid freezing
            stride = 100
            
            for i in range(0, len(self._3d_picking_vertex_list), stride):
                xw, yw, zw = self._3d_picking_vertex_list[i]
                try:
                    winx, winy, winz = gluProject(xw, yw, zw, model_np, proj_np, viewport)
                except Exception:
                    continue
                
                winy = viewport[3] - winy
                dx = winx - mx
                dy = winy - my
                d2 = dx * dx + dy * dy
                if d2 < best_dist2:
                    best_dist2 = d2
                    best_idx = i # Map back to original index

            if best_idx is None or best_dist2 > (32 * 32):
                return

            # --- Start of new debug code ---
            # Remove previous marker
            if self._debug_marker is not None:
                try:
                    self.view3d_widget.removeItem(self._debug_marker)
                except Exception:
                    pass # Ignore errors if item is already gone

            # Get the original 3D coordinate of the picked vertex
            picked_vertex_orig = self._3d_picking_vertex_list[best_idx]

            # Create a new marker
            # We need to transform this original vertex just like the main mesh is transformed
            # so it shows up in the right place.
            center = self._3d_transform_params['center']
            scale = self._3d_transform_params['scale']
            z_boost = self._3d_transform_params['z_boost']
            model_name = self._3d_transform_params['model_name']
            affine_A = self._3d_transform_params['affine_A']
            affine_t = self._3d_transform_params['affine_t']

            marker_vert = np.array(picked_vertex_orig, dtype=float)
            transformed_marker_vert = (marker_vert - center) * scale
            if transformed_marker_vert.shape[0] > 2:
                transformed_marker_vert[2] *= z_boost
            if model_name == 'harmonic_entropy' and affine_A is not None and affine_t is not None:
                transformed_xy = (transformed_marker_vert[:2] @ affine_A.T) + affine_t
                transformed_marker_vert[:2] = transformed_xy

            self._debug_marker = gl.GLScatterPlotItem(pos=np.array([transformed_marker_vert]), size=15, color=(1, 0, 0, 1), pxMode=False)
            self.view3d_widget.addItem(self._debug_marker)
            # --- End of new debug code ---

            grid_idx = self._3d_index_map[best_idx]
            r, c = grid_idx
            bank_name = self.bank_order[self.current_bank_index]
            if bank_name not in self.data_banks or self.data_banks[bank_name] is None:
                return
            X, Y, Z = self.data_banks[bank_name]
            xval, yval = float(X[r, c]), float(Y[r, c])
            
            c2 = yval * 2 / np.sqrt(3)
            c1 = xval - (c2 / 2)

            self._update_sound_from_cents(c1, c2)

        except Exception as e:
            print(f"Error in _pick_and_play_3d: {e}")
            return

    def _on_view3d_mouse_press(self, event):
        self.isohe_widget.dragging = True
        self._pick_and_play_3d(event.pos())

    def _on_view3d_mouse_move(self, event):
        if self.isohe_widget.dragging:
            self._pick_and_play_3d(event.pos())

    def _on_view3d_mouse_release(self, event):
        self.isohe_widget.dragging = False
        self.isohe_widget.stop_sound()

    def _build_3d_mesh(self, model_name):
        # Build a GL mesh from X/Y/Z arrays and add to view3d_widget
        if self.view3d_widget is None or gl is None:
            return
        try:
            X, Y, Z = self.data_banks[model_name]

            if model_name in ['harmonic_entropy', 'sethares']:
                Z = gaussian_filter(Z, sigma=1.5)

            rows, cols = X.shape
            verts = []
            index_map = {}
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if not np.isnan(Z[r, c]):
                        verts.append([float(X[r, c]), float(Y[r, c]), float(Z[r, c])])
                        index_map[(r, c)] = idx
                        idx += 1

            if len(verts) == 0:
                return

            self._3d_picking_vertex_list = list(verts) # Store original vertices for picking
            np_verts = np.array(verts, dtype=float)
            
            # normalize vertices so the mesh fits nicely inside the GL view
            v_min = np.nanmin(np_verts, axis=0)
            v_max = np.nanmax(np_verts, axis=0)
            center = (v_min + v_max) / 2.0
            extent = v_max - v_min
            # Compute scale from X/Y extents only so boosting Z later doesn't change XY proportions
            xy_extent = max(extent[0], extent[1]) if extent.size >= 2 else np.max(extent)
            if xy_extent == 0 or not np.isfinite(xy_extent):
                scale = 1.0
            else:
                # target a comfortable size in view coordinates
                target_size = 1.6
                scale = target_size / xy_extent
            np_verts_transformed = (np_verts - center) * scale

            # Boost Z-axis to give the mesh more depth so it matches the exported .obj appearance
            z_boost = 12.0
            if model_name == 'sethares':
                z_boost *= 12.0
            if np_verts_transformed.shape[1] > 2:
                np_verts_transformed[:, 2] = np_verts_transformed[:, 2] * z_boost

            # Store transformation parameters
            self._3d_transform_params = {
                'center': center,
                'scale': scale,
                'z_boost': z_boost,
                'model_name': model_name,
                'affine_A': None,
                'affine_t': None,
                'data_X': X, # Store original X, Y for Z lookup
                'data_Y': Y,
                'data_Z': Z
            }

            # If this is the Harmonic Entropy model, ensure the XY shape is an equilateral triangle
            # by computing an affine transform from the detected corner triangle -> canonical equilateral triangle.
            if model_name == 'harmonic_entropy':
                pts = np_verts_transformed[:, :2]
                # Monotone chain convex hull implementation
                def _convex_hull(points):
                    pts_sorted = sorted(map(tuple, points.tolist()))
                    if len(pts_sorted) <= 1:
                        return pts_sorted
                    def cross(o, a, b):
                        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                    lower = []
                    for p in pts_sorted:
                        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                            lower.pop()
                        lower.append(p)
                    upper = []
                    for p in reversed(pts_sorted):
                        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                            upper.pop()
                        upper.append(p)
                    hull = lower[:-1] + upper[:-1]
                    return hull

                hull = _convex_hull(pts)
                if len(hull) >= 3:
                    # choose the triangle of hull vertices with maximal area
                    hull_arr = np.array(hull)
                    n = len(hull_arr)
                    best_area = 0.0
                    best_tri = None
                    for i in range(n):
                        for j in range(i+1, n):
                            for k in range(j+1, n):
                                a = hull_arr[i]
                                b = hull_arr[j]
                                c = hull_arr[k]
                                area = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) * 0.5
                                if area > best_area:
                                    best_area = area
                                    best_tri = (a, b, c)

                    if best_tri is not None and best_area > 1e-12:
                        src = np.array(best_tri)
                        # compute centroid and radius to construct canonical equilateral triangle
                        src_centroid = src.mean(axis=0)
                        src_r = np.mean(np.linalg.norm(src - src_centroid, axis=1))
                        # equilateral triangle vertices (pointing up) relative to centroid
                        r = src_r
                        tri_target = np.array([
                            [0.0, r],
                            [-np.sqrt(3)/2 * r, -0.5 * r],
                            [ np.sqrt(3)/2 * r, -0.5 * r]
                        ])
                        # translate target to have same centroid as source
                        tgt_centroid = tri_target.mean(axis=0)
                        tri_target = tri_target - tgt_centroid + src_centroid

                        # Solve affine transform A (2x2) and t s.t. A @ src_i + t = tgt_i
                        S = np.hstack([src, np.ones((3,1))])  # 3x3
                        T = tri_target  # 3x2
                        # Solve for 3x2 matrix M where S @ M = T, M contains [A|t]
                        try:
                            M, *_ = np.linalg.lstsq(S, T, rcond=None)
                            A = M[:2, :].T
                            t = M[2, :]
                            # apply transform
                            transformed_xy = (pts @ A.T) + t
                            np_verts_transformed[:, :2] = transformed_xy
                            self._3d_transform_params['affine_A'] = A
                            self._3d_transform_params['affine_t'] = t
                        except Exception:
                            pass

            faces = []
            for r in range(rows - 1):
                for c in range(cols - 1):
                    try:
                        v1 = index_map.get((r, c))
                        v2 = index_map.get((r + 1, c))
                        v3 = index_map.get((r + 1, c + 1))
                        v4 = index_map.get((r, c + 1))
                        if v1 is not None and v2 is not None and v3 is not None:
                            faces.append([v1, v2, v3])
                        if v1 is not None and v3 is not None and v4 is not None:
                            faces.append([v1, v3, v4])
                    except Exception:
                        continue

            md = MeshData(vertexes=np_verts_transformed, faces=np.array(faces))
            # Attempt to compute per-vertex colors from the Z axis using the
            # same colormap used by the 2D topo views so the 3D mesh shares
            # the same gradient coloring.
            try:
                vert_colors = None
                if np_verts_transformed.shape[1] > 2:
                    z_vals = np_verts_transformed[:, 2]
                    z_min = float(np.nanmin(z_vals))
                    z_max = float(np.nanmax(z_vals))
                    if np.isfinite(z_min) and np.isfinite(z_max):
                        if (z_max - z_min) == 0:
                            norm = np.zeros_like(z_vals)
                        else:
                            norm = (z_vals - z_min) / (z_max - z_min)
                        rgba = self.custom_cm(norm)
                        vert_colors = np.array(rgba, dtype=float)
                        # Try attaching colors to the MeshData (preferred)
                        try:
                            md.setVertexColors(vert_colors)
                        except Exception:
                            # Some MeshData versions may not support setVertexColors;
                            # we'll rely on GLMeshItem reading colors from meshdata when present
                            pass
            except Exception:
                pass
            # remove old mesh
            if hasattr(self, '_3d_mesh_item') and self._3d_mesh_item is not None:
                try:
                    self.view3d_widget.removeItem(self._3d_mesh_item)
                except Exception:
                    pass

            blue_color = (4/255.0, 55/255.0, 242/255.0, 1.0)
            # Create a shiny mesh if our ShinyMesh subclass is available; otherwise fall back to GLMeshItem
            if _ShinyMeshClass is not None:
                try:
                                        mesh_item = _ShinyMeshClass(meshdata=md, smooth=True, drawEdges=False, shader='shaded', color=blue_color, specular=(1.0, 1.0, 1.0), shininess=128.0)
                except Exception:
                    mesh_item = GLMeshItem(meshdata=md, smooth=True, drawEdges=False, shader='shaded', color=blue_color)
            else:
                mesh_item = GLMeshItem(meshdata=md, smooth=True, drawEdges=False, shader='shaded', color=blue_color)
            # ensure mesh is centered in the GLView and default to a bird's-eye framing
            # First prefer an orthographic projection when available
            try:
                if hasattr(self.view3d_widget, 'setProjection'):
                    try:
                        self.view3d_widget.setProjection('ortho')
                    except Exception:
                        # some pyqtgraph versions may raise; fall back to opts
                        try:
                            self.view3d_widget.opts['projection'] = 'ortho'
                        except Exception:
                            pass
            except Exception:
                pass

            # Birdseye parameters (adjustable)
            birdseye_elevation = 60
            birdseye_azimuth = 0
            birdseye_distance = 2.2

            def _apply_camera():
                # Try positional API, then keyword API, then write opts directly.
                try:
                    self.view3d_widget.setCameraPosition(QVector3D(0.0, 0.0, 0.0), birdseye_distance, birdseye_elevation, birdseye_azimuth)
                    return
                except Exception:
                    pass
                try:
                    self.view3d_widget.setCameraPosition(center=QVector3D(0.0, 0.0, 0.0), distance=birdseye_distance, elevation=birdseye_elevation, azimuth=birdseye_azimuth)
                    return
                except Exception:
                    pass
                try:
                    self.view3d_widget.opts['center'] = QVector3D(0.0, 0.0, 0.0)
                    self.view3d_widget.opts['distance'] = birdseye_distance
                    self.view3d_widget.opts['elevation'] = birdseye_elevation
                    self.view3d_widget.opts['azimuth'] = birdseye_azimuth
                    try:
                        self.view3d_widget.opts['projection'] = 'ortho'
                    except Exception:
                        pass
                    try:
                        self.view3d_widget.update()
                    except Exception:
                        pass
                except Exception:
                    pass

            # apply immediately and schedule a short delayed re-apply to handle timing issues
            try:
                _apply_camera()
                QTimer.singleShot(50, _apply_camera)
            except Exception:
                pass

            self.view3d_widget.addItem(mesh_item)
            self._3d_mesh_item = mesh_item

            # build reverse mapping idx -> (r,c)
            rev_map = {}
            for (rc, ind) in index_map.items():
                rev_map[ind] = rc
            self._3d_index_map = rev_map

            # Add EDO dots if enabled
            if self.mode_button.isChecked():
                self._add_edo_dots_to_3d_view()

        except Exception as e:
            print(f"Error building 3D mesh: {e}")
            return

    def _add_edo_dots_to_3d_view(self):
        if self.view3d_widget is None or gl is None or not self._3d_transform_params:
            return

        # Clear existing EDO dots
        if self._3d_edo_dots_item is not None:
            try:
                self.view3d_widget.removeItem(self._3d_edo_dots_item)
            except Exception:
                pass
            self._3d_edo_dots_item = None

        try:
            edo = int(self.main_app.edo_entry.text())
            if edo <= 0: return
        except (ValueError, AttributeError):
            return

        equave_ratio = self.isohe_widget.equave
        equave_cents = 1200 * np.log2(float(equave_ratio))
        step_in_cents = 1200 / edo

        # Get transformation parameters
        center = self._3d_transform_params['center']
        scale = self._3d_transform_params['scale']
        z_boost = self._3d_transform_params['z_boost']
        model_name = self._3d_transform_params['model_name']
        affine_A = self._3d_transform_params['affine_A']
        affine_t = self._3d_transform_params['affine_t']
        data_X = self._3d_transform_params['data_X']
        data_Y = self._3d_transform_params['data_Y']
        data_Z = self._3d_transform_params['data_Z']

        # Get image dimensions from the Z data array
        height, width = data_Z.shape
        max_cents_x = 1200 * np.log2(float(equave_ratio))
        max_cents_y = 1200 * np.log2(float(equave_ratio)) * np.sqrt(3) / 2 # This is from triangle_generator.py

        edo_points_3d = []

        num_steps = int(round(equave_cents / step_in_cents))

        for i in range(num_steps + 1):
            for j in range(num_steps + 1 - i):
                c1_edo = i * step_in_cents
                c2_edo = j * step_in_cents

                if c1_edo + c2_edo > equave_cents + 1e-9:
                    continue

                # Use the same formula as generate_triangle_image for cx, cy
                cx_data = c1_edo + (c2_edo / 2)
                cy_data = c2_edo * np.sqrt(3) / 2

                # Convert cents coordinates to pixel coordinates (similar to triangle_generator.py)
                # Adjust scaling to map to width/height instead of width-1/height-1
                c_idx = int(round((cx_data / max_cents_x) * (width - 1)))
                r_idx = int(round((cy_data / max_cents_y) * (height - 1)))

                # Clamp indices to ensure they are within bounds
                r_idx = max(0, min(r_idx, height - 1))
                c_idx = max(0, min(c_idx, width - 1))

                # Only add the point if data_Z is not NaN at this location
                if not np.isnan(data_Z[r_idx, c_idx]):
                    z_val = data_Z[r_idx, c_idx]
                    
                    # Create a temporary vertex for transformation
                    temp_vert = np.array([data_X[r_idx, c_idx], data_Y[r_idx, c_idx], z_val], dtype=float)

                    # Apply the same normalization and scaling as the mesh
                    transformed_vert = (temp_vert - center) * scale

                    # Apply Z-boost
                    transformed_vert[2] = transformed_vert[2] * z_boost

                    # Apply Harmonic Entropy Affine Transform if applicable
                    if model_name == 'harmonic_entropy' and affine_A is not None and affine_t is not None:
                        transformed_xy = (transformed_vert[:2] @ affine_A.T) + affine_t
                        transformed_vert[:2] = transformed_xy

                    # Adjust Z-offset to 0.0
                    transformed_vert[2] += 0.0

                    edo_points_3d.append(transformed_vert)

                    # Generate label if labels button is checked
                    if self.labels_button.isChecked():
                        label_text = f"[0, {i}, {i+j}]"

                        # Get edo value
                        try:
                            edo = int(self.main_app.edo_entry.text())
                        except (ValueError, AttributeError):
                            edo = 12 # Default to 12 EDO if not available

                        # Use the helper function from isohe_widget
                        font_size = self.isohe_widget._get_font_size_for_label(label_text, dots_mode='EDO', edo_value=edo)
                        
                        # Position the label slightly above the dot if dots are shown,
                        # otherwise centered at the dot's position.
                        label_pos = transformed_vert.copy()
                        if self.dots_button.isChecked(): # If dots are shown
                            label_pos[2] += 0.05 # Slightly higher than the dot
                        else:
                            # When dots are not shown, center the label.
                            # GLTextItem doesn't have a direct way to get text height,
                            # so we'll use a small fixed offset for vertical centering in 3D.
                            label_pos[2] += 0.01 # A small offset to prevent Z-fighting with the plane

                        text_item = gl.GLTextItem(pos=label_pos, text=label_text, color=(1.0, 1.0, 1.0, 1.0), font_size=font_size)
                        self._3d_edo_labels_items.append(text_item)

        if edo_points_3d:
            edo_points_3d = np.array(edo_points_3d)
            # Use a small size and grey color for the dots
            self._3d_edo_dots_item = gl.GLScatterPlotItem(pos=edo_points_3d, size=0.025, color=(0.5, 0.5, 0.5, 1.0), pxMode=False)
            self.view3d_widget.addItem(self._3d_edo_dots_item)
        
        # Add labels if labels button is checked
        if self.labels_button.isChecked():
            for item in self._3d_edo_labels_items:
                self.view3d_widget.addItem(item)

    def save_triangle_image(self):
        if self.view3d_widget and self.view3d_widget.isVisible():
            file_path, _ = QFileDialog.getSaveFileName(self, "Save 3D View", "", "PNG Files (*.png)")
            if not file_path:
                return
            
            image = QImage(self.view3d_widget.size(), QImage.Format_ARGB32)
            image.fill(Qt.transparent)
            painter = QPainter(image)
            self.view3d_widget.render(painter)
            painter.end()
            
            image.save(file_path)
            return

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
            
            self.worker = SetharesWorker(spectrum_data, ref_freq, max_interval, 0.01, 400, 1.5, 0)
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
