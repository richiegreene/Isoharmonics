import numpy as np
import pygame
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPolygonF, QBrush, QColor, QPen, QFont, QPainterPath
from fractions import Fraction
from audio.generators import generate_combined_playback_buffer
from theory.calculations import format_series_segment, calculate_edo_step
from theory.notation.engine import calculate_single_note
from utils.formatters import to_subscript

class IsoHEWidget(QWidget):
    def __init__(self, main_app=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_app = main_app
        self.cents_label = {}
        self.note_labels = {}
        self.current_ratios = [1, 1, 1]
        self.buffer_duration = 0.75  # seconds (new default)
        self.fade_duration = 0.5    # seconds (new default)
        self.echo_gain = 0.5         # new default
        self.last_played_ratios = None
        self.stationary_update_interval = 200  # ms, default value
        self.stationary_timer = QTimer(self)
        self.stationary_timer.setInterval(self.stationary_update_interval)
        self.stationary_timer.timeout.connect(self.stationary_update_sound)
        self.update_dimensions()

        self.pivot_voice = "lower"  # Default pivot
        self.pivot_pitch = 0.0
        self.pitch1 = 0.0
        self.pitch2 = 0.0
        self.pitch3 = 0.0

        # No extra layout needed; triangle/svg will fill window

        self.drag_timer = QTimer(self)
        self.drag_timer.setSingleShot(True)
        self.drag_timer.timeout.connect(self.on_drag_timeout)
        self.last_drag_point = None
        self.sound = None
        self.triangle_image = None

    def set_pivot_voice(self, voice):
        if self.pivot_voice != voice:
            if voice == "upper":
                self.pivot_pitch = self.pitch3
            elif voice == "middle":
                self.pivot_pitch = self.pitch2
            else:  # lower
                self.pivot_pitch = self.pitch1
            self.pivot_voice = voice

    def update_ratios_label(self):
        from theory.calculations import format_series_segment
        from gui.widgets.draggable_fraction_line_edit import DraggableFractionLineEdit
        # Map each ratio to nearest 95-odd-limit interval
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
            # Position above cursor
            x = self.last_drag_point.x()
            y = self.last_drag_point.y() - 24
            painter.drawText(QPointF(x, y), ratios_str)
            
    def on_drag_timeout(self):
        # This method is called when the drag_timer times out. Implement logic as needed.
        pass

    def set_equave(self, equave):
        self.equave = equave
        self.update()

    def set_triangle_image(self, image):
        self.triangle_image = image
        self.update()

    def update_dimensions(self):
        padding = 30
        side_length = min(self.width(), self.height()) - 2 * padding
        height = (np.sqrt(3) / 2) * side_length
        
        self.v1 = QPointF(self.width() / 2, padding) # Top vertex
        self.v2 = QPointF(self.width() / 2 - side_length / 2, padding + height) # Bottom-left
        self.v3 = QPointF(self.width() / 2 + side_length / 2, padding + height) # Bottom-right
        self.triangle = QPolygonF([self.v1, self.v2, self.v3])
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.white, 1, Qt.SolidLine))

        if self.triangle_image:
            rect = self.triangle.boundingRect().toRect()
            path = QPainterPath()
            path.addPolygon(self.triangle)
            painter.setClipPath(path)
            painter.drawImage(rect, self.triangle_image)
            painter.setClipping(False)
        else:
            painter.setBrush(QBrush(QColor(11, 6, 86)))
            painter.drawPolygon(self.triangle)

        font = QFont("Arial Nova", 12)
        painter.setFont(font)
        painter.setPen(Qt.white)

        equave_ratio = self.equave
        top_corner_ratio = [1, 1, equave_ratio]
        bottom_right_ratio = [1, equave_ratio, equave_ratio]

        painter.drawText(self.v1 + QPointF(-15, -15), format_series_segment(top_corner_ratio))
        # Vertically align both bottom labels with the bottom side of the triangle
        bottom_y = self.v2.y()  # Both corners share the same y
        painter.drawText(QPointF(self.v2.x() - 30, bottom_y), "1:1:1")
        painter.drawText(QPointF(self.v3.x() + 5, bottom_y), format_series_segment(bottom_right_ratio))

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
        # Simulate minimal cursor movement in a small circle to avoid stationary pulsing
        if self.dragging and self.last_drag_point is not None:
            # Parameters for minimal movement
            radius = 0.01  # pixels
            if not hasattr(self, 'stationary_angle'):
                self.stationary_angle = 0.0
            self.stationary_angle += 0.2  # radians per update
            # Calculate new position in a small circle around the last_drag_point
            dx = radius * np.cos(self.stationary_angle)
            dy = radius * np.sin(self.stationary_angle)
            moved_point = QPointF(self.last_drag_point.x() + dx, self.last_drag_point.y() + dy)
            self.update_ratios_and_sound(moved_point)
            # Do not update self.last_drag_point, so the circle stays centered

    def update_ratios_and_sound(self, pos):
        if not self.triangle.containsPoint(pos, Qt.OddEvenFill):
            print("Mouse outside triangle, skipping update.")
            return

        p = np.array([pos.x(), pos.y()])
        v1 = np.array([self.v1.x(), self.v1.y()])
        v2 = np.array([self.v2.x(), self.v2.y()])
        v3 = np.array([self.v3.x(), self.v3.y()])

        detT = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1])
        if abs(detT) < 1e-8:
            print("Degenerate triangle: detT is zero.")
            return

        w1 = ((v2[1] - v3[1]) * (p[0] - v3[0]) + (v3[0] - v2[0]) * (p[1] - v3[1])) / detT
        w2 = ((v3[1] - v1[1]) * (p[0] - v3[0]) + (v1[0] - v3[0]) * (p[1] - v3[1])) / detT
        w3 = 1.0 - w1 - w2

        w = np.array([w1, w2, w3])
        w[w < 0] = 0
        if w.sum() == 0:
            print("All barycentric weights are zero.")
            return
        w /= w.sum()

        # Triangular mapping
        equave_cents = 1200 * np.log2(float(self.equave))
        cx = w[2] * equave_cents
        cy = w[0] * equave_cents

        # p is the absolute pitch of the pivot voice in cents
        p = self.pivot_pitch

        if self.pivot_voice == "upper":
            self.pitch1 = p - cx - cy
            self.pitch2 = p - cy
            self.pitch3 = p
        elif self.pivot_voice == "middle":
            self.pitch1 = p - cx
            self.pitch2 = p
            self.pitch3 = p + cy
        else:  # lower
            self.pitch1 = p
            self.pitch2 = p + cx
            self.pitch3 = p + cx + cy

        # Convert back to ratios for sound generation
        def cents_to_ratio(c):
            return 2 ** (c / 1200)

        r1 = cents_to_ratio(self.pitch1)
        r2 = cents_to_ratio(self.pitch2)
        r3 = cents_to_ratio(self.pitch3)
        new_ratios = [r1, r2, r3]
        
        # --- Display Logic ---
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
            if error_str in ["+0", "-0"]:
                error_str = ""
            self.note_labels[str(i + 1)].setText(f"{note_name_with_octave} {error_str}")

        # 2. Ratio Display: Format the ratios based on their absolute values.
        self.current_ratios = new_ratios
        self.update_ratios_label()

        # --- Sound Generation ---
        if not all(np.isfinite(new_ratios)) or any(r <= 0 for r in new_ratios):
            print("Invalid ratios computed:", new_ratios)
            return

        if self.last_played_ratios is None or any(abs(a - b) > 1e-6 for a, b in zip(new_ratios, self.last_played_ratios)):
            self.update_sound()
            self.last_played_ratios = list(new_ratios)

    def update_sound(self):
        if not self.dragging:
            return

        old_sound = self.sound

        try:
            # Get the first isoharmonic ratio from the main window's table
            first_iso_item = self.main_app.table.item(1, 1)
            if first_iso_item:
                first_iso_ratio = Fraction(first_iso_item.text())
                base_freq = float(first_iso_ratio) * 261.6
            else:
                base_freq = 261.6 # Fallback
        except (ValueError, ZeroDivisionError):
            base_freq = 261.6

        if len(self.current_ratios) != 3 or any(r <= 0 or not np.isfinite(r) for r in self.current_ratios):
            print("Invalid current_ratios for sound:", self.current_ratios)
            return

        triangle_freqs = [base_freq * r for r in self.current_ratios]
        if any(f <= 0 or not np.isfinite(f) for f in triangle_freqs):
            print("Invalid triangle frequencies:", triangle_freqs)
            return

        timbre = getattr(self.main_app.visualizer, "current_timbre", None)
        roll_off = getattr(self.main_app, "roll_off_rate", 0.0)
        phase = getattr(self.main_app, "phase_factor", 0.0)
        duration = 0.7
        if not (0.01 <= duration <= 10.0):
            print("Invalid duration:", duration)
            return

        all_frequencies = []
        all_ratios = []

        if timbre and 'ratios' in timbre:
            timbre_ratios = timbre['ratios']
            for base_f, tri_ratio in zip(triangle_freqs, self.current_ratios):
                for t_ratio in timbre_ratios:
                    freq = base_f * float(t_ratio)
                    ratio = float(tri_ratio * t_ratio)
                    if freq > 0 and np.isfinite(freq):
                        all_frequencies.append(freq)
                        all_ratios.append(ratio)
        else:
            all_frequencies = triangle_freqs
            all_ratios = self.current_ratios

        triangle_freqs = [base_freq * r for r in self.current_ratios]
        if any(f <= 0 or not np.isfinite(f) for f in triangle_freqs):
            print("Invalid triangle frequencies:", triangle_freqs)
            return

        # Restore timbre logic and continuous playback
        timbre = getattr(self.main_app.visualizer, "current_timbre", None)
        roll_off = getattr(self.main_app, "roll_off_rate", 0.0)
        phase = getattr(self.main_app, "phase_factor", 0.0)
        duration = self.buffer_duration
        all_frequencies = []
        all_ratios = []

        if timbre and 'ratios' in timbre:
            timbre_ratios = timbre['ratios']
            for base_f, tri_ratio in zip(triangle_freqs, self.current_ratios):
                for t_ratio in timbre_ratios:
                    freq = base_f * float(t_ratio)
                    ratio = float(tri_ratio * t_ratio)
                    if freq > 0 and np.isfinite(freq):
                        all_frequencies.append(freq)
                        all_ratios.append(ratio)
        else:
            # Chorus effect: duplicate each voice with slight detuning
            detune_cents = [0, 1, -1]
            for freq in triangle_freqs:
                for cents in detune_cents:
                    detuned_freq = freq * (2 ** (cents / 1200))
                    all_frequencies.append(detuned_freq)
            all_ratios = [1] * len(all_frequencies)

        try:
            buffer = generate_combined_playback_buffer(
                all_frequencies, all_ratios, duration, roll_off, phase
            )
            if buffer is None or buffer.size == 0:
                print("Generated buffer is empty.")
                return

            # Fade-in/out envelope
            fade_len = int(self.fade_duration * buffer.shape[0])
            if fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                buffer[:fade_len] = (buffer[:fade_len].T * fade_in).T
                buffer[-fade_len:] = (buffer[-fade_len:].T * fade_out).T

            # Optional: add echo/chorus delay
            echo_delay = int(0.03 * buffer.shape[0])
            echo_gain = self.echo_gain
            if echo_delay > 0:
                echo_buf = np.zeros_like(buffer)
                echo_buf[echo_delay:] = (buffer[:-echo_delay] * echo_gain).astype(buffer.dtype)
                buffer = np.clip(buffer + echo_buf, -32768, 32767)

            # Cross-fade: start new sound, fade out old sound
            self.sound = pygame.sndarray.make_sound(buffer)
            self.sound.play(loops=-1)
            if old_sound:
                try:
                    old_sound.fadeout(int(self.fade_duration * 1000))
                except Exception as e:
                    print("Exception during fadeout:", e)
        except Exception as e:
            print("Exception during sound generation:", e)
            self.sound = None

    def stop_sound(self):
        if self.sound:
            try:
                self.sound.fadeout(int(self.fade_duration * 1000))
            except Exception as e:
                print("Exception during fadeout:", e)
            self.sound = None
