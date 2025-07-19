from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt
import math
import pygame
from audio.generators import generate_combined_playback_buffer
from audio.playback import stop_sound
from fractions import Fraction
from theory.calculations import simplify_ratio
from audio.playback import play_single_sine_wave

class VisualizerWidget(QWidget):
    def __init__(self, isoharmonic_entry, main_app, parent=None):
        super().__init__(parent)
        self.isoharmonic_entry = isoharmonic_entry
        self.main_app = main_app
        self.series = []
        self.fundamental = Fraction("1/1")
        self.setMinimumHeight(100)
        self.hovered_ratio = None
        self.setMouseTracking(True)
        self.current_sounds = []
        self.duration = 2
        self.current_timbre = None
        self.ji_timbre = None
        self.edo_timbre = None

        self.dragging_fundamental = False
        self.dragging_isoharmonic = False
        self.start_mouse_x = 0
        self.start_fundamental_log_ratio = 0.0
        self.start_isoharmonic_log_ratio = 0.0
        self.drag_sensitivity = 0.005 # Adjust this value as needed
        self.drag_active = False
        self.current_sound = None # To hold the currently playing sound for draggable dots

        self.last_click_time = 0 # For double-click detection
        self.double_click_interval = 250 # milliseconds

        self.pressed_dot_type = None
        self.pressed_dot_ratio = None
        self.initial_mouse_pos = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor('#0b0656'), 4)
        painter.setPen(pen)
        line_y = self.height() - 30
        painter.drawLine(10, line_y, self.width() - 10, line_y)
        if self.series:
            lowest_iso = min(self.series)
            current_ratio = lowest_iso
            self.grey_dots = []
            while current_ratio > 0:
                current_ratio = current_ratio - self.fundamental
                if current_ratio > 0:
                    try:
                        freq = 261.6 * float(Fraction(simplify_ratio(current_ratio)))
                        x = self.freq_to_x(freq)
                        self.grey_dots.append((x, line_y, current_ratio))
                        painter.setBrush(QColor("#808080"))
                        painter.drawEllipse(x - 6, line_y - 6, 12, 12)
                    except Exception:
                        break
            self.white_dots = []
            for ratio in self.series:
                try:
                    freq = 261.6 * float(Fraction(simplify_ratio(ratio)))
                    x = self.freq_to_x(freq)
                    self.white_dots.append((x, line_y, ratio))
                    painter.setBrush(Qt.white)
                    painter.drawEllipse(x - 6, line_y - 6, 12, 12)
                except Exception:
                    continue
            try:
                freq_fundamental = 261.6 * float(Fraction(simplify_ratio(self.fundamental)))
                x_fundamental = self.freq_to_x(freq_fundamental)
                self.blue_dot = (x_fundamental, line_y, self.fundamental)
                painter.setBrush(QColor("#0b0656"))
                painter.drawEllipse(x_fundamental - 10, line_y - 10, 20, 20)
            except Exception:
                pass
            try:
                isoharmonic_ratio = Fraction(self.isoharmonic_entry.text())
                freq_isoharmonic = 261.6 * float(isoharmonic_ratio)
                x_isoharmonic = self.freq_to_x(freq_isoharmonic)
                painter.setBrush(QColor("#0437f2"))
                painter.drawEllipse(x_isoharmonic - 10, line_y - 10, 20, 20)
            except Exception:
                pass
            if self.hovered_ratio:
                x, y, ratio = self.hovered_ratio
                ratio_str = simplify_ratio(ratio)
                painter.setPen(Qt.white)
                painter.setFont(self.font())
                text_width = painter.fontMetrics().width(ratio_str)
                painter.drawText(x - text_width // 2, y - 10, ratio_str)

    def freq_to_x(self, freq):
        min_freq = 8.1758
        max_freq = 8372.018
        log_min = math.log10(min_freq)
        log_max = math.log10(max_freq)
        log_freq = math.log10(freq)
        return int((log_freq - log_min) / (log_max - log_min) * (self.width() - 20)) + 10

    def mouseMoveEvent(self, event):
        mouse_x = event.x()
        mouse_y = event.y()
        tolerance = 10

        

        if self.dragging_fundamental:
            delta_x = event.x() - self.start_mouse_x
            new_log_ratio = self.start_fundamental_log_ratio + delta_x * self.drag_sensitivity
            new_ratio = 10**new_log_ratio
            
            # Update the fundamental_entry QLineEdit
            try:
                self.main_app.fundamental_entry.setText(str(Fraction(new_ratio).limit_denominator(1000)))
            except ValueError:
                pass # Keep the old value if conversion fails
            return

        if self.dragging_isoharmonic:
            delta_x = event.x() - self.start_mouse_x
            new_log_ratio = self.start_isoharmonic_log_ratio + delta_x * self.drag_sensitivity
            new_ratio = 10**new_log_ratio

            # Update the isoharmonic_entry QLineEdit
            try:
                self.isoharmonic_entry.setText(str(Fraction(new_ratio).limit_denominator(1000)))
            except ValueError:
                pass # Keep the old value if conversion fails
            return

        # Original hover logic
        for x, y, ratio in self.grey_dots + self.white_dots + [self.blue_dot]:
            if abs(mouse_x - x) < tolerance and abs(mouse_y - y) < tolerance:
                self.hovered_ratio = (x, y, ratio)
                self.update()
                return
        if self.hovered_ratio:
            self.hovered_ratio = None
            self.update()

    def mousePressEvent(self, event):
        mouse_x = event.x()
        mouse_y = event.y()
        tolerance = 15 # Increased tolerance for easier clicking on larger dots
        
        self.drag_active = False # Reset drag_active at the start of a press
        self.initial_mouse_pos = event.pos()

        current_time = event.timestamp() # Get current click time

        # Check for fundamental dot (dark blue)
        if hasattr(self, 'blue_dot') and self.blue_dot:
            x_fundamental, y_fundamental, ratio_fundamental = self.blue_dot
            if abs(mouse_x - x_fundamental) < tolerance and abs(mouse_y - y_fundamental) < tolerance:
                self.pressed_dot_type = 'fundamental'
                self.pressed_dot_ratio = ratio_fundamental
                if (current_time - self.last_click_time) < self.double_click_interval:
                    # Double-click detected, activate drag
                    self.stop_all_sounds()
                    self.drag_active = True
                    self.dragging_fundamental = True
                    self.start_mouse_x = self.initial_mouse_pos.x()
                    try:
                        self.start_fundamental_log_ratio = math.log10(float(self.main_app.fundamental_entry.text()))
                    except ValueError:
                        self.start_fundamental_log_ratio = 0.0
                else:
                    # Single click, play sound
                    self._play_fundamental_sound()
                self.last_click_time = current_time
                return

        # Check for isoharmonic dot (light blue)
        if hasattr(self, 'isoharmonic_entry') and self.isoharmonic_entry:
            try:
                isoharmonic_ratio = Fraction(self.isoharmonic_entry.text())
                freq_isoharmonic = 261.6 * float(isoharmonic_ratio)
                x_isoharmonic = self.freq_to_x(freq_isoharmonic)
                
                if abs(mouse_x - x_isoharmonic) < tolerance and abs(mouse_y - (self.height() - 30)) < tolerance:
                    self.pressed_dot_type = 'isoharmonic'
                    self.pressed_dot_ratio = isoharmonic_ratio
                    if (current_time - self.last_click_time) < self.double_click_interval:
                        # Double-click detected, activate drag
                        self.stop_all_sounds()
                        self.drag_active = True
                        self.dragging_isoharmonic = True
                        self.start_mouse_x = self.initial_mouse_pos.x()
                        try:
                            self.start_isoharmonic_log_ratio = math.log10(float(self.isoharmonic_entry.text()))
                        except ValueError:
                            self.start_isoharmonic_log_ratio = 0.0
                    else:
                        # Single click, play sound
                        self._play_isoharmonic_sound()
                    self.last_click_time = current_time
                    return
            except Exception:
                pass # Ignore if isoharmonic_entry is invalid

        # If not a draggable dot, proceed with sound playback for other dots
        for x, y, ratio in self.grey_dots + self.white_dots:
            if abs(mouse_x - x) < tolerance and abs(mouse_y - y) < tolerance:
                freq = 261.6 * float(Fraction(simplify_ratio(ratio)))
                if self.current_timbre:
                    frequencies = [freq * float(r) for r in self.current_timbre['ratios']]
                    ratios = self.current_timbre['ratios']
                    roll_off = self.current_timbre['roll_off']
                    phase = self.current_timbre['phase']
                    
                    pygame.mixer.init()
                    buffer = generate_combined_playback_buffer(
                        frequencies, 
                        ratios, 
                        self.duration, 
                        roll_off,
                        phase
                    )
                    sound = pygame.sndarray.make_sound(buffer)
                    sound.play()
                    self.current_sounds.append(sound)
                else:
                    sound = play_single_sine_wave(freq, self.duration)
                    self.current_sounds = [sound]
                return

    def mouseReleaseEvent(self, event):
        # Stop any currently playing sound from dragging
        if self.current_sound:
            self.current_sound.stop()
            self.current_sound = None

        self.dragging_fundamental = False
        self.dragging_isoharmonic = False
        self.drag_active = False
        self.pressed_dot_type = None
        self.pressed_dot_ratio = None
        self.initial_mouse_pos = None
        super().mouseReleaseEvent(event)

    

    def stop_all_sounds(self):
        for sound in self.current_sounds:
            stop_sound(sound)
        self.current_sounds = []

    def _activate_drag(self):
        self.stop_all_sounds()
        self.drag_active = True
        if self.pressed_dot_type == 'fundamental':
            self.dragging_fundamental = True
            self.start_mouse_x = self.initial_mouse_pos.x()
            try:
                self.start_fundamental_log_ratio = math.log10(float(self.main_app.fundamental_entry.text()))
            except ValueError:
                self.start_fundamental_log_ratio = 0.0
        elif self.pressed_dot_type == 'isoharmonic':
            self.dragging_isoharmonic = True
            self.start_mouse_x = self.initial_mouse_pos.x()
            try:
                self.start_isoharmonic_log_ratio = math.log10(float(self.isoharmonic_entry.text()))
            except ValueError:
                self.start_isoharmonic_log_ratio = 0.0
        self.pressed_dot_type = None
        self.pressed_dot_ratio = None
        self.initial_mouse_pos = None

    def _play_fundamental_sound(self):
        if hasattr(self, 'blue_dot') and self.blue_dot:
            freq = 261.6 * float(Fraction(simplify_ratio(self.blue_dot[2])))
            if self.current_timbre:
                frequencies = [freq * float(r) for r in self.current_timbre['ratios']]
                ratios = self.current_timbre['ratios']
                roll_off = self.current_timbre['roll_off']
                phase = self.current_timbre['phase']
                pygame.mixer.init()
                buffer = generate_combined_playback_buffer(
                    frequencies,
                    ratios,
                    self.duration,
                    roll_off,
                    phase
                )
                sound = pygame.sndarray.make_sound(buffer)
                sound.play()
                self.current_sounds.append(sound)
            else:
                sound = play_single_sine_wave(freq, self.duration)
                self.current_sounds = [sound]

    def _play_isoharmonic_sound(self):
        if hasattr(self, 'isoharmonic_entry') and self.isoharmonic_entry:
            try:
                isoharmonic_ratio = Fraction(self.isoharmonic_entry.text())
                freq = 261.6 * float(isoharmonic_ratio)
                if self.current_timbre:
                    frequencies = [freq * float(r) for r in self.current_timbre['ratios']]
                    ratios = self.current_timbre['ratios']
                    roll_off = self.current_timbre['roll_off']
                    phase = self.current_timbre['phase']
                    pygame.mixer.init()
                    buffer = generate_combined_playback_buffer(
                        frequencies,
                        ratios,
                        self.duration,
                        roll_off,
                        phase
                    )
                    sound = pygame.sndarray.make_sound(buffer)
                    sound.play()
                    self.current_sounds.append(sound)
                else:
                    sound = play_single_sine_wave(freq, self.duration)
                    self.current_sounds = [sound]
            except Exception:
                pass # Ignore if isoharmonic_entry is invalid
