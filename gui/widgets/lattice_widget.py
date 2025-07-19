from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QTimer
import math
from fractions import Fraction
import pygame
from theory.calculations import generate_iso_series
from theory.notation.engine import calculate_single_note
from audio.playback import play_single_sine_wave
from audio.generators import generate_combined_playback_buffer

class LatticeWidget(QWidget):
    def __init__(self, main_app, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.setMouseTracking(True)
        self.hovered_node = None
        self.nodes = []
        self.hex_size = 10
        self.setMinimumSize(600, 400)
        self.label_font = QFont("Arial Nova")
        self.label_font.setPointSize(12)
        self.equave_steps = set()
        self.c_steps = set()

        self.mouse_press_timer = QTimer(self)
        self.mouse_press_timer.setInterval(1000)  # 1 second
        self.mouse_press_timer.setSingleShot(True)
        self.mouse_press_timer.timeout.connect(self._handle_long_press)

        self.is_dragging = False
        self.drag_start_position = None
        self.pressed_node_value = None
        self.current_sound = None

    def update_grid(self):
        if self.main_app.lattice_window is None:
            return

        self.nodes = []
        self.equave_steps.clear()
        self.c_steps.clear()

        if self.main_app.lattice_window.is_edo_mode:
            steps = []
            for row in range(self.main_app.table.rowCount()):
                item = self.main_app.table.item(row, 4)
                if item and item.text().strip():
                    try:
                        steps.append(int(item.text()))
                    except ValueError:
                        pass

            if not steps:
                return

            e_start = int(self.main_app.lattice_window.equave_start.text() or "1")
            e_end = int(self.main_app.lattice_window.equave_end.text() or "2")
            h_start = int(self.main_app.lattice_window.h_start.text() or "2")
            h_end = int(self.main_app.lattice_window.h_end.text() or "3")
            d_start = int(self.main_app.lattice_window.d_start.text() or "4")
            d_end = int(self.main_app.lattice_window.d_end.text() or "5")

            if any(i < 0 or i >= len(steps) for i in [e_start, e_end, h_start, h_end, d_start, d_end]):
                raise IndexError("Invalid partial index")

            equave_interval = steps[e_end] - steps[e_start]
            horizontal_interval = steps[h_end] - steps[h_start]
            diagonal_interval = steps[d_end] - steps[d_start]

            grid_radius = 12
            center_step = 0
            nodes = []
            equave_steps = equave_interval if self.main_app.lattice_window.equave_toggle.isChecked() else 0

            for q in range(-grid_radius, grid_radius + 1):
                for r in range(-grid_radius, grid_radius + 1):
                    s = -q - r
                    if abs(s) > grid_radius:
                        continue

                    step = center_step + (q * horizontal_interval) + (r * diagonal_interval)

                    if equave_steps > 0:
                        step %= equave_steps
                        if step < 0:
                            step += equave_steps

                    nodes.append((q, r, step))

            self.nodes = nodes

            edo = int(self.main_app.edo_entry.text())

            if equave_interval > 0:
                self.equave_steps = {i * equave_interval % equave_interval for i in range(edo // equave_interval + 1)}
            self.c_steps = {i * edo for i in range(-grid_radius, grid_radius + 1)}

        else:
            fundamental = Fraction(self.main_app.fundamental_entry.text())
            isoharmonic = Fraction(self.main_app.isoharmonic_entry.text())
            partials_above = int(self.main_app.partials_above_entry.text())
            partials_below = int(self.main_app.partials_below_entry.text())

            series = generate_iso_series(fundamental, isoharmonic, partials_above, partials_below)

            if len(series) < 5:
                return

            try:
                h_start = int(self.main_app.lattice_window.h_start.text() or "2") - 1
                h_end = int(self.main_app.lattice_window.h_end.text() or "3") - 1
                d_start = int(self.main_app.lattice_window.d_start.text() or "4") - 1
                d_end = int(self.main_app.lattice_window.d_end.text() or "5") - 1

                if any(i < 0 or i >= len(series) for i in [h_start, h_end, d_start, d_end]):
                    raise IndexError

                horizontal_interval = series[h_end] / series[h_start]
                diagonal_interval = series[d_start] / series[d_end]

            except Exception as e:
                print("Invalid custom lattice distance:", e)
                self.nodes = []
                self.update()
                return

            grid_radius = 12
            center_ratio = Fraction(self.main_app.isoharmonic_entry.text())
            nodes = []

            for q in range(-grid_radius, grid_radius + 1):
                for r in range(-grid_radius, grid_radius + 1):
                    s = -q - r
                    if abs(s) > grid_radius:
                        continue
                    ratio = center_ratio * (horizontal_interval ** q) * (diagonal_interval ** -r)
                    if self.main_app.lattice_window.equave_toggle.isChecked():
                        try:
                            e_start = int(self.main_app.lattice_window.equave_start.text()) - 1
                            e_end = int(self.main_app.lattice_window.equave_end.text()) - 1
                            series = generate_iso_series(fundamental, isoharmonic, partials_above, partials_below)

                            if any(i < 0 or i >= len(series) for i in [e_start, e_end]):
                                raise IndexError

                            equave = series[e_end] / series[e_start]

                            while ratio >= equave:
                                ratio /= equave
                            while ratio < 1:
                                ratio *= equave

                        except Exception as e:
                            print("Equave reduction error:", e)
                    nodes.append((q, r, ratio))
            self.nodes = nodes


        self.update()

    def calculate_layout(self):
        width = self.width()
        height = self.height()

        max_q = max(abs(node[0]) for node in self.nodes) if self.nodes else 0
        max_r = max(abs(node[1]) for node in self.nodes) if self.nodes else 0

        hex_width = 2 * self.hex_size
        hex_height = math.sqrt(3) * self.hex_size

        required_width = (max_q * 1.5 + max_r * 0.75) * hex_width
        required_height = (max_r + max_q * 0.5) * hex_height

        width_scale = width / required_width if required_width > 0 else 1
        height_scale = height / required_height if required_height > 0 else 1
        self.scale = min(width_scale, height_scale) * 2

        self.origin_x = width / 2
        self.origin_y = height / 2

    def axial_to_pixel(self, q, r):
        x = self.hex_size * self.scale * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
        y = self.hex_size * self.scale * (-3 / 2 * r)
        return (self.origin_x + x, self.origin_y + y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self.nodes:
            painter.setPen(Qt.gray)
            painter.drawText(self.rect(), Qt.AlignCenter, "n/a")
            return

        self.calculate_layout()
        drawn_edges = set()

        painter.setPen(QPen(QColor('#0b0656'), 3))
        for node in self.nodes:
            q, r, value = node
            x1, y1 = self.axial_to_pixel(q, r)

            neighbors = [
                (q + 1, r),
                (q + 1, r - 1),
                (q, r - 1),
                (q - 1, r),
                (q - 1, r + 1),
                (q, r + 1)
            ]

            for nq, nr in neighbors:
                if (nq, nr) in [(n[0], n[1]) for n in self.nodes]:
                    edge = tuple(sorted([(q, r), (nq, nr)]))
                    if edge not in drawn_edges:
                        x2, y2 = self.axial_to_pixel(nq, nr)
                        painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                        drawn_edges.add(edge)

        visible_nodes = []
        for node in self.nodes:
            q, r, value = node
            x, y = self.axial_to_pixel(q, r)
            visible_nodes.append((x, y, value))

            is_equave = value in self.equave_steps
            is_c = value in self.c_steps

            if is_equave:
                size = 20
                color = QColor('#0437f2')
            elif is_c:
                size = 20
                color = QColor('#0b0656')
            else:
                is_center = (value == 0 if self.main_app.lattice_window.is_edo_mode else value == Fraction(self.main_app.isoharmonic_entry.text()))
                size = 16 if is_center else 12
                color = QColor('#0437f2') if is_center else QColor('#A0A0A0')

            painter.setBrush(color)
            painter.drawEllipse(int(x - size/2), int(y - size/2), size, size)

        if self.hovered_node:
            hx, hy, value = self.hovered_node
            painter.setFont(self.label_font)
            painter.setPen(Qt.white)

            if self.main_app.lattice_window.is_edo_mode:
                edo = int(self.main_app.edo_entry.text())
                step = value % edo
                note_name = calculate_single_note(step, edo)
                text_width = painter.fontMetrics().width(note_name)
                painter.drawText(int(hx - text_width / 2), int(hy - 15), note_name)
            else:
                ratio_str = f"{value.numerator}/{value.denominator}"
                text_width = painter.fontMetrics().width(ratio_str)
                painter.drawText(int(hx - text_width / 2), int(hy - 15), ratio_str)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        closest = None
        min_dist = float('inf')

        for node in self.nodes:
            q, r, value = node
            x, y = self.axial_to_pixel(q, r)
            dx = pos.x() - x
            dy = pos.y() - y
            dist = dx * dx + dy * dy

            if dist < 100 and dist < min_dist:
                min_dist = dist
                closest = (x, y, value)

        self.hovered_node = closest
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.hovered_node:
            self.pressed_node_value = self.hovered_node[2]
            self.drag_start_position = event.pos()
            self.mouse_press_timer.start()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.mouse_press_timer.isActive():
                self.mouse_press_timer.stop()
                # This was a short click, play sound
                if self.hovered_node:
                    self._play_node_sound(self.hovered_node[2])
            elif self.is_dragging:
                self.is_dragging = False
                self.pressed_node_value = None
                self.drag_start_position = None
                if self.current_sound:
                    self.current_sound.stop()
                    self.current_sound = None

    def _handle_long_press(self):
        # Long press detected, stop sound and enable dragging
        if self.current_sound:
            self.current_sound.stop()
            self.current_sound = None
        self.is_dragging = True

    def _play_node_sound(self, value):
        if self.main_app.lattice_window.is_edo_mode:
            edo = int(self.main_app.edo_entry.text())
            freq = 261.6 * (2 ** (value / edo))
        else:
            freq = 261.6 * float(value)

        if self.main_app.visualizer.current_timbre:
            frequencies = [freq * float(r) for r in self.main_app.visualizer.current_timbre['ratios']]
            ratios = self.main_app.visualizer.current_timbre['ratios']
            roll_off = self.main_app.roll_off_rate
            phase = self.main_app.phase_factor

            buffer = generate_combined_playback_buffer(
                frequencies, ratios,
                self.main_app.visualizer.duration,
                roll_off, phase
            )
            self.current_sound = pygame.sndarray.make_sound(buffer)
            self.current_sound.play()
        else:
            self.current_sound = play_single_sine_wave(freq, self.main_app.visualizer.duration)

    def resizeEvent(self, event):
        self.calculate_layout()
        self.update()
        super().resizeEvent(event)
