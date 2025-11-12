import numpy as np
from PyQt5.QtWidgets import QSlider, QLineEdit
import pygame
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSpinBox
from PyQt5.QtCore import pyqtSignal, QTimer, QThread
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPolygonF, QBrush, QColor, QPen, QFont, QPainterPath
from PyQt5.QtSvg import QSvgGenerator
from fractions import Fraction
from audio.generators import generate_combined_playback_buffer
from theory.calculations import format_series_segment, calculate_edo_step, generate_ji_triads
from theory.notation.engine import calculate_single_note
from utils.formatters import to_subscript
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

class IsoHEWidget(QWidget):
    def __init__(self, main_app=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_app = main_app
        self.cents_label = {}
        self.note_labels = {}
        self.current_ratios = [1, 1, 1]
        self.buffer_duration = 0.75
        self.fade_duration = 0.5
        self.echo_gain = 0.5
        self.last_played_ratios = None
        self.stationary_update_interval = 200
        self.stationary_timer = QTimer(self)
        self.stationary_timer.setInterval(self.stationary_update_interval)
        self.stationary_timer.timeout.connect(self.stationary_update_sound)
        self.update_dimensions()

        self.pivot_voice = "lower"
        self.pivot_pitch = 0.0
        self.pitch1 = 0.0
        self.pitch2 = 0.0
        self.pitch3 = 0.0

        self.drag_timer = QTimer(self)
        self.drag_timer.setSingleShot(True)
        self.drag_timer.timeout.connect(self.on_drag_timeout)
        self.last_drag_point = None
        self.sound = None
        self.triangle_image = None
        self.show_dots = False
        self.show_labels = False
        self.dots_mode = 'JI'
        self.odd_limit = 15

        self.topo_data = None
        self.topo_colormap = None
        self.topo_model_name = None
        # playback worker handle
        self._playback_worker = None

    def set_dots_mode(self, mode):
        self.dots_mode = mode
        self.update()

    def set_odd_limit(self, limit):
        self.odd_limit = limit
        self.update()

    def set_topo_data(self, data, colormap, model_name):
        self.topo_data = data
        self.topo_colormap = colormap
        self.topo_model_name = model_name
        self.update()

    def clear_topo_data(self):
        self.topo_data = None
        self.topo_colormap = None
        self.topo_model_name = None
        self.update()

    def set_show_dots(self, show):
        self.show_dots = show
        self.update()

    def set_show_labels(self, show):
        self.show_labels = show
        self.update()

    def set_pivot_voice(self, voice):
        if self.pivot_voice != voice:
            if voice == "upper":
                self.pivot_pitch = self.pitch3
            elif voice == "middle":
                self.pivot_pitch = self.pitch2
            else:
                self.pivot_pitch = self.pitch1
            self.pivot_voice = voice

    def update_ratios_label(self):
        from theory.calculations import format_series_segment
        from gui.widgets.draggable_fraction_line_edit import DraggableFractionLineEdit
        intervals = DraggableFractionLineEdit().generate_95_odd_limit_intervals()
        def nearest_interval(val):
            return min(intervals, key=lambda x: abs(float(x) - float(val)))

        simplified = [nearest_interval(r) for r in self.current_ratios]
        ratios_str = format_series_segment(simplified)
        self.ratios_label.setText(ratios_str)
        
    def draw_ratios_above_cursor(self, painter):
        if self.dragging and self.last_drag_point is not None:
            from theory.calculations import format_series_segment
            ratios_str = format_series_segment(self.current_ratios)
            font = QFont("Arial Nova", 14)
            painter.setFont(font)
            painter.setPen(Qt.white)
            x = self.last_drag_point.x()
            y = self.last_drag_point.y() - 24
            painter.drawText(QPointF(x, y), ratios_str)
            
    def on_drag_timeout(self):
        pass

    def set_equave(self, equave):
        self.equave = equave
        self.update()

    def set_triangle_image(self, image):
        self.triangle_image = image
        self.clear_topo_data()
        self.update()

    def update_dimensions(self):
        padding = 30
        side_length = min(self.width(), self.height()) - 2 * padding
        height = (np.sqrt(3) / 2) * side_length
        
        self.v1 = QPointF(self.width() / 2, padding)
        self.v2 = QPointF(self.width() / 2 - side_length / 2, padding + height)
        self.v3 = QPointF(self.width() / 2 + side_length / 2, padding + height)
        self.triangle = QPolygonF([self.v1, self.v2, self.v3])
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        self.paint_widget(painter)

    def paint_widget(self, painter, for_svg=False):
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background or image
        if self.topo_data and self.topo_colormap and self.topo_model_name:
            self.paint_topo_contours(painter, self.topo_data, self.topo_colormap, self.topo_model_name)
        elif self.triangle_image:
            rect = self.triangle.boundingRect().toRect()
            path = QPainterPath()
            path.addPolygon(self.triangle)
            painter.setClipPath(path)
            painter.drawImage(rect, self.triangle_image)
            painter.setClipping(False)
        else:
            painter.setBrush(QBrush(QColor(11, 6, 86)))
            painter.drawPolygon(self.triangle)

        # Always draw dots and labels, regardless of whether it's for SVG or not
        painter.setPen(QPen(Qt.white, 1, Qt.SolidLine))
        if self.show_dots or self.show_labels: # Draw if either dots or labels are enabled
            self.draw_dots(painter)

    def save_svg(self, file_path, topo_data=None, colormap=None, model_name=None):
        if not file_path:
            return

        svg_generator = QSvgGenerator()
        svg_generator.setFileName(file_path)
        svg_generator.setSize(self.size())
        svg_generator.setViewBox(self.rect())
        svg_generator.setTitle("Isoharmonic Triad")
        svg_generator.setDescription("An SVG depiction of the isoharmonic triad display.")

        painter = QPainter()
        painter.begin(svg_generator)
        
        # Temporarily set topo data for SVG generation if provided
        original_topo_data = (self.topo_data, self.topo_colormap, self.topo_model_name)
        if topo_data and colormap and model_name:
            self.set_topo_data(topo_data, colormap, model_name)
        
        self.paint_widget(painter, for_svg=True)

        # Restore original topo data
        self.set_topo_data(*original_topo_data)

        painter.end()

    def paint_topo_contours(self, painter, topo_data, colormap, model_name):
        X, Y, Z = topo_data
        levels = 30
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        if np.isnan(z_min) or np.isnan(z_max):
            return
        contour_levels = np.linspace(z_min, z_max, levels)

        fig, ax = plt.subplots()
        if model_name == 'harmonic_entropy':
            contours = ax.contour(X, Y, Z, levels=contour_levels, cmap=colormap, linewidths=1.4, origin='lower')
        elif model_name == 'sethares':
            contours = ax.contour(X, Y, Z, levels=contour_levels, cmap=colormap, linewidths=1.4)
        else:
            plt.close(fig)
            return
        
        widget_triangle_rect = self.triangle.boundingRect()
        x_min_data, x_max_data = np.nanmin(X), np.nanmax(X)
        y_min_data, y_max_data = np.nanmin(Y), np.nanmax(Y)

        path = QPainterPath()
        path.addPolygon(self.triangle)
        painter.setClipPath(path)

        for collection in contours.collections:
            color = collection.get_edgecolor()[0]
            qt_color = QColor.fromRgbF(color[0], color[1], color[2], color[3])
            pen = QPen(qt_color, 1.4)
            painter.setPen(pen)

            for path_mpl in collection.get_paths():
                q_path = QPainterPath()
                for i, (vertex, code) in enumerate(path_mpl.iter_segments()):
                    x_norm = (vertex[0] - x_min_data) / (x_max_data - x_min_data)
                    y_norm = 1 - ((vertex[1] - y_min_data) / (y_max_data - y_min_data))

                    x_widget = widget_triangle_rect.x() + x_norm * widget_triangle_rect.width()
                    y_widget = widget_triangle_rect.y() + y_norm * widget_triangle_rect.height()
                    
                    if i == 0:
                        q_path.moveTo(x_widget, y_widget)
                    else:
                        q_path.lineTo(x_widget, y_widget)
                painter.drawPath(q_path)
        
        painter.setClipping(False)
        plt.close(fig)

    def draw_dots(self, painter):
        if self.dots_mode == 'JI':
            self.draw_ji_dots(painter)
        else:
            self.draw_edo_dots(painter)

    def _get_font_size_for_label(self, label_text, dots_mode, edo_value=12):
        base_size_ji = 10 # Font size for num_digits <= 3 (as per user's latest instruction)
        min_font_size_ji = 0.025  # Minimum font size (as per user's latest instruction)
        base_size_edo = 10 # Base size for EDO labels

        if dots_mode == 'JI':
            # JI sizing logic based on total number of digits in the label, with exponential decay
            num_digits = sum(c.isdigit() for c in label_text)
            
            # Exponential decay: font size halves roughly every 3 additional digits
            # decay_factor = 0.5**(1/3) approx 0.7937
            decay_factor = 0.79370046571 # More precise value for 0.5^(1/3)

            # Ensure num_digits is at least 3 to avoid negative exponent or division by zero issues
            # and to ensure base_size_ji is applied for the smallest labels.
            effective_num_digits = max(3, num_digits)

            calculated_font_size = base_size_ji * (decay_factor ** (effective_num_digits - 3))
            return max(min_font_size_ji, calculated_font_size)

        elif dots_mode == 'EDO':
            # EDO sizing logic with cap
            if edo_value <= 0:
                return base_size_edo # Fallback for invalid EDO
            
            scaling_factor = (12 / edo_value)
            # Ensure fonts do not increase in size from the default
            scaling_factor = min(1.0, scaling_factor) # Cap at 1.0
            
            return base_size_edo * scaling_factor
        return base_size_edo # Default fallback

    def draw_ji_dots(self, painter):
        try:
            limit = self.odd_limit
            if limit <= 0: return
        except (ValueError, AttributeError):
            return

        equave_cents = 1200 * np.log2(float(self.equave))
        
        painter.setBrush(QColor('#A0A0A0'))
        painter.setPen(QPen(QColor('#0437f2'), 1))

        triads = generate_ji_triads(limit, self.equave)

        for (cx, cy), label in triads:
            if cx + cy > equave_cents + 1e-9: continue

            w1 = cy / equave_cents
            w3 = cx / equave_cents
            w2 = 1.0 - w1 - w3

            if not (-1e-9 <= w1 <= 1 + 1e-9 and -1e-9 <= w2 <= 1 + 1e-9 and -1e-9 <= w3 <= 1 + 1e-9): continue

            x = w1 * self.v1.x() + w2 * self.v2.x() + w3 * self.v3.x()
            y = w1 * self.v1.y() + w2 * self.v2.y() + w3 * self.v3.y()

            if self.show_dots:
                painter.drawEllipse(QPointF(x, y), 4, 4)

            if self.show_labels:
                font_size = self._get_font_size_for_label(label, self.dots_mode)
                font = QFont("Arial Nova", font_size)
                painter.setFont(font)
                painter.setPen(Qt.white)
                text_width = painter.fontMetrics().width(label)
                text_height = painter.fontMetrics().height()
                
                if self.show_dots:
                    # Position slightly above the dot
                    painter.drawText(QPointF(x - text_width / 2, y - 8), label)
                else:
                    # Position centered where the dot would be
                    painter.drawText(QPointF(x - text_width / 2, y + text_height / 4), label)

    def draw_edo_dots(self, painter):
        try:
            edo = int(self.main_app.edo_entry.text())
            if edo <= 0: return
        except (ValueError, AttributeError):
            return

        equave_cents = 1200 * np.log2(float(self.equave))
        step_in_cents = 1200 / edo

        painter.setBrush(QColor('#A0A0A0'))
        painter.setPen(QPen(QColor('#0437f2'), 1))

        num_steps = int(equave_cents / step_in_cents)

        for i in range(num_steps + 1):
            for j in range(num_steps + 1 - i):
                cx = i * step_in_cents
                cy = j * step_in_cents

                if cx + cy > equave_cents: continue

                w1 = cy / equave_cents
                w3 = cx / equave_cents
                w2 = 1.0 - w1 - w3

                if not (0 <= w1 <= 1 and 0 <= w2 <= 1 and 0 <= w3 <= 1): continue

                x = w1 * self.v1.x() + w2 * self.v2.x() + w3 * self.v3.x()
                y = w1 * self.v1.y() + w2 * self.v2.y() + w3 * self.v3.y()

                if self.show_dots:
                    painter.drawEllipse(QPointF(x, y), 4, 4)

                if self.show_labels:
                    label = f"[0, {i}, {i+j}]"
                    edo = int(self.main_app.edo_entry.text()) # Get edo value here
                    font_size = self._get_font_size_for_label(label, self.dots_mode, edo_value=edo)
                    font = QFont("Arial Nova", font_size)
                    painter.setFont(font)
                    painter.setPen(Qt.white)
                    text_width = painter.fontMetrics().width(label)
                    text_height = painter.fontMetrics().height()

                    if self.show_dots:
                        # Position slightly above the dot
                        painter.drawText(QPointF(x - text_width / 2, y - 8), label)
                    else:
                        # Position centered where the dot would be
                        painter.drawText(QPointF(x - text_width / 2, y + text_height / 4), label)

    def mousePressEvent(self, event):
        if self.triangle.containsPoint(event.pos(), Qt.OddEvenFill):
            self.dragging = True
            self.update_ratios_and_sound(event.pos())
            self.stationary_timer.start()
            self.last_drag_point = event.pos()
        else:
            self.dragging = False
            self.stationary_timer.stop()

    def mouseMoveEvent(self, event):
        if self.dragging:
            if self.triangle.containsPoint(event.pos(), Qt.OddEvenFill):
                self.update_ratios_and_sound(event.pos())
                self.last_drag_point = event.pos()
            else:
                self.dragging = False
                self.stationary_timer.stop()
                self.stop_sound()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.stationary_timer.stop()
        self.stop_sound()
        
    def stationary_update_sound(self):
        if self.dragging and self.last_drag_point is not None:
            radius = 0.01
            if not hasattr(self, 'stationary_angle'):
                self.stationary_angle = 0.0
            self.stationary_angle += 0.2
            dx = radius * np.cos(self.stationary_angle)
            dy = radius * np.sin(self.stationary_angle)
            moved_point = QPointF(self.last_drag_point.x() + dx, self.last_drag_point.y() + dy)
            self.update_ratios_and_sound(moved_point)

    def update_ratios_and_sound(self, pos):
        if not self.triangle.containsPoint(pos, Qt.OddEvenFill):
            return

        p = np.array([pos.x(), pos.y()])
        v1 = np.array([self.v1.x(), self.v1.y()])
        v2 = np.array([self.v2.x(), self.v2.y()])
        v3 = np.array([self.v3.x(), self.v3.y()])

        detT = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1])
        if abs(detT) < 1e-8: return

        w1 = ((v2[1] - v3[1]) * (p[0] - v3[0]) + (v3[0] - v2[0]) * (p[1] - v3[1])) / detT
        w2 = ((v3[1] - v1[1]) * (p[0] - v3[0]) + (v1[0] - v3[0]) * (p[1] - v3[1])) / detT
        w3 = 1.0 - w1 - w2

        w = np.array([w1, w2, w3])
        w[w < 0] = 0
        if w.sum() == 0: return
        w /= w.sum()

        equave_cents = 1200 * np.log2(float(self.equave))
        cx = w[2] * equave_cents
        cy = w[0] * equave_cents

        p_pitch = self.pivot_pitch

        if self.pivot_voice == "upper":
            self.pitch1, self.pitch2, self.pitch3 = p_pitch - cx - cy, p_pitch - cy, p_pitch
        elif self.pivot_voice == "middle":
            self.pitch1, self.pitch2, self.pitch3 = p_pitch - cx, p_pitch, p_pitch + cy
        else:
            self.pitch1, self.pitch2, self.pitch3 = p_pitch, p_pitch + cx, p_pitch + cx + cy

        def cents_to_ratio(c): return 2 ** (c / 1200)
        new_ratios = [cents_to_ratio(p) for p in [self.pitch1, self.pitch2, self.pitch3]]
        
        self.cents_label["3"].setText(f"{int(round(self.pitch3))}")
        self.cents_label["2"].setText(f"{int(round(self.pitch2))}")
        self.cents_label["1"].setText(f"{int(round(self.pitch1))}")

        edo = int(self.main_app.edo_entry.text())
        for i, pitch in enumerate([self.pitch1, self.pitch2, self.pitch3]):
            step_str, error = calculate_edo_step(pitch, edo)
            step = int(step_str.replace("-", "-"))
            note_name = calculate_single_note(step, edo)
            octave = 4 + (pitch // 1200)
            note_name_with_octave = note_name + to_subscript(int(octave))
            error_str = f"{round(-error):+}".replace("-", "-")
            if error_str in ["+0", "-0"]: error_str = ""
            self.note_labels[str(i + 1)].setText(f"{note_name_with_octave} {error_str}")

        self.current_ratios = new_ratios
        self.update_ratios_label()

        if not all(np.isfinite(new_ratios)) or any(r <= 0 for r in new_ratios): return
        if self.last_played_ratios is None or any(abs(a - b) > 1e-6 for a, b in zip(new_ratios, self.last_played_ratios)):
            self.update_sound()
            self.last_played_ratios = list(new_ratios)

    def update_sound(self):
        if not self.dragging: return
        old_sound = self.sound
        try:
            first_iso_item = self.main_app.table.item(1, 1)
            base_freq = float(Fraction(first_iso_item.text())) * 261.6 if first_iso_item else 261.6
        except (ValueError, ZeroDivisionError):
            base_freq = 261.6

        if len(self.current_ratios) != 3 or any(r <= 0 or not np.isfinite(r) for r in self.current_ratios): return
        triangle_freqs = [base_freq * r for r in self.current_ratios]
        if any(f <= 0 or not np.isfinite(f) for f in triangle_freqs): return

        timbre = getattr(self.main_app.visualizer, "current_timbre", None)
        roll_off = getattr(self.main_app, "roll_off_rate", 0.0)
        phase = getattr(self.main_app, "phase_factor", 0.0)
        duration = self.buffer_duration
        all_frequencies, all_ratios = [], []

        if timbre and 'ratios' in timbre:
            for base_f, tri_ratio in zip(triangle_freqs, self.current_ratios):
                for t_ratio in timbre['ratios']:
                    freq, ratio = base_f * float(t_ratio), float(tri_ratio * t_ratio)
                    if freq > 0 and np.isfinite(freq):
                        all_frequencies.append(freq)
                        all_ratios.append(ratio)
        else:
            for freq in triangle_freqs:
                for cents in [0, 1, -1]:
                    all_frequencies.append(freq * (2 ** (cents / 1200)))
            all_ratios = [1] * len(all_frequencies)

        try:
            buffer = generate_combined_playback_buffer(all_frequencies, all_ratios, duration, roll_off, phase)
            if buffer is None or buffer.size == 0: return

            fade_len = int(self.fade_duration * buffer.shape[0])
            if fade_len > 0:
                fade_in, fade_out = np.linspace(0, 1, fade_len), np.linspace(1, 0, fade_len)
                buffer[:fade_len] = (buffer[:fade_len].T * fade_in).T
                buffer[-fade_len:] = (buffer[-fade_len:].T * fade_out).T

            echo_delay = int(0.03 * buffer.shape[0])
            if echo_delay > 0:
                echo_buf = np.zeros_like(buffer)
                echo_buf[echo_delay:] = (buffer[:-echo_delay] * self.echo_gain).astype(buffer.dtype)
                buffer = np.clip(buffer + echo_buf, -32768, 32767)

            self.sound = pygame.sndarray.make_sound(buffer)
            self.sound.play(loops=-1)
            if old_sound: old_sound.fadeout(int(self.fade_duration * 1000))
        except Exception as e:
            self.sound = None

    # Non-blocking background playback: compute buffer in a QThread, then play on main thread
    class _PlaybackWorker(QThread):
        finished = pyqtSignal(object)

        def __init__(self, all_frequencies, all_ratios, duration, roll_off, phase):
            super().__init__()
            self.all_frequencies = all_frequencies
            self.all_ratios = all_ratios
            self.duration = duration
            self.roll_off = roll_off
            self.phase = phase

        def run(self):
            try:
                buf = generate_combined_playback_buffer(self.all_frequencies, self.all_ratios, self.duration, self.roll_off, self.phase)
                self.finished.emit(buf)
            except Exception:
                self.finished.emit(None)

    def play_current_ratios_background(self):
        """Compute the playback buffer in a background thread and play it when ready.
        This avoids blocking the GUI during heavy buffer generation.
        """
        # Prepare frequencies and ratios similar to update_sound
        try:
            first_iso_item = self.main_app.table.item(1, 1)
            base_freq = float(Fraction(first_iso_item.text())) * 261.6 if first_iso_item else 261.6
        except Exception:
            base_freq = 261.6

        if len(self.current_ratios) != 3 or any(r <= 0 or not np.isfinite(r) for r in self.current_ratios): return
        triangle_freqs = [base_freq * r for r in self.current_ratios]
        if any(f <= 0 or not np.isfinite(f) for f in triangle_freqs): return

        timbre = getattr(self.main_app.visualizer, "current_timbre", None)
        roll_off = getattr(self.main_app, "roll_off_rate", 0.0)
        phase = getattr(self.main_app, "phase_factor", 0.0)
        duration = self.buffer_duration

        all_frequencies, all_ratios = [], []
        if timbre and 'ratios' in timbre:
            for base_f, tri_ratio in zip(triangle_freqs, self.current_ratios):
                for t_ratio in timbre['ratios']:
                    freq, ratio = base_f * float(t_ratio), float(tri_ratio * t_ratio)
                    if freq > 0 and np.isfinite(freq):
                        all_frequencies.append(freq)
                        all_ratios.append(ratio)
        else:
            for freq in triangle_freqs:
                for cents in [0, 1, -1]:
                    all_frequencies.append(freq * (2 ** (cents / 1200)))
            all_ratios = [1] * len(all_frequencies)

        # stop any existing worker
        try:
            if self._playback_worker is not None and self._playback_worker.isRunning():
                self._playback_worker.terminate()
        except Exception:
            pass

        # start new worker
        worker = IsoHEWidget._PlaybackWorker(all_frequencies, all_ratios, duration, roll_off, phase)

        def _on_finished(buf):
            try:
                if buf is None or buf.size == 0:
                    return
                old_sound = self.sound
                fade_len = int(self.fade_duration * buf.shape[0])
                if fade_len > 0:
                    fade_in, fade_out = np.linspace(0, 1, fade_len), np.linspace(1, 0, fade_len)
                    buf[:fade_len] = (buf[:fade_len].T * fade_in).T
                    buf[-fade_len:] = (buf[-fade_len:].T * fade_out).T

                echo_delay = int(0.03 * buf.shape[0])
                if echo_delay > 0:
                    echo_buf = np.zeros_like(buf)
                    echo_buf[echo_delay:] = (buf[:-echo_delay] * self.echo_gain).astype(buf.dtype)
                    buf = np.clip(buf + echo_buf, -32768, 32767)

                self.sound = pygame.sndarray.make_sound(buf)
                self.sound.play(loops=-1)
                if old_sound:
                    try:
                        old_sound.fadeout(int(self.fade_duration * 1000))
                    except Exception:
                        pass
            except Exception:
                self.sound = None

        worker.finished.connect(_on_finished)
        self._playback_worker = worker
        worker.start()

    def stop_sound(self):
        if self.sound:
            try:
                self.sound.fadeout(int(self.fade_duration * 1000))
            except Exception: pass
            self.sound = None
