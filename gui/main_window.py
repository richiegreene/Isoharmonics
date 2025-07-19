from PyQt5.QtWidgets import QFileDialog, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QSizePolicy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
import numpy as np
import pygame
import os
import unicodedata
import wave
from fractions import Fraction
from gui.widgets.waveform_canvas import WaveformCanvas
from gui.widgets.draggable_fraction_line_edit import DraggableFractionLineEdit
from gui.widgets.draggable_decimal_line_edit import DraggableDecimalLineEdit
from gui.widgets.draggable_integer_line_edit import DraggableIntegerLineEdit, ReversedDraggableIntegerLineEdit
from gui.widgets.copyable_label import CopyableLabel
from gui.widgets.visualizer_widget import VisualizerWidget
from gui.lattice_window import LatticeWindow
from gui.isohe_window import IsoHEWindow
from audio.playback import play_sine_wave, stop_all_sounds, stop_sound
from audio.generators import generate_combined_playback_buffer, normalization_factor, max_amplitude, sample_rate, fade_duration
from theory.calculations import ratio_to_cents, calculate_edo_step, generate_iso_series, format_series_segment, simplify_ratio
from theory.edo import assign_12edo_notation
from theory.notation.engine import calculate_single_note

class IsoharmonicApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.active_button = None
        self.active_playbacks = {}
        self.current_sound = None
        self.active_timers = {}
        self.current_channel = None
        self.current_edo_channel = None
        self.setWindowTitle("Harmonics")
        self.setGeometry(100, 100, 1250, 650)
        
        self.apply_dark_theme()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_panel.setLayout(self.left_layout)
        self.layout.addWidget(self.left_panel, stretch=1)
        
        self.fundamental_label = QLabel("Virtual Fundamental")
        self.fundamental_entry = DraggableFractionLineEdit("1/1")
        self.fundamental_entry.setFont(QFont("Arial Nova", 14))
        self.fundamental_entry.textChanged.connect(self.update_fundamental_tuner_readout)
        self.fundamental_entry.focusIn.connect(self.safe_stop_sounds)
        self.fundamental_tuner_readout_label = QLineEdit("C₄")
        self.fundamental_tuner_readout_label.setReadOnly(True)
        self.fundamental_tuner_readout_label.setFont(QFont("Arial Nova", 14))
        self.left_layout.addWidget(self.fundamental_label)
        self.left_layout.addWidget(self.fundamental_entry)
        self.left_layout.addWidget(self.fundamental_tuner_readout_label)
        
        self.isoharmonic_label = QLabel("Isoharmonic Partial")
        self.isoharmonic_entry = DraggableFractionLineEdit("1")
        self.isoharmonic_entry.setFont(QFont("Arial Nova", 14))
        self.isoharmonic_entry.textChanged.connect(self.update_isoharmonic_tuner_readout)
        self.isoharmonic_entry.focusIn.connect(self.safe_stop_sounds)
        self.isoharmonic_tuner_readout_label = QLineEdit("C₄")
        self.isoharmonic_tuner_readout_label.setReadOnly(True)
        self.isoharmonic_tuner_readout_label.setFont(QFont("Arial Nova", 14))
        self.left_layout.addWidget(self.isoharmonic_label)
        self.left_layout.addWidget(self.isoharmonic_entry)
        self.left_layout.addWidget(self.isoharmonic_tuner_readout_label)
        
        self.partials_above_label = QLabel("Above Partials Listed")
        self.partials_above_entry = DraggableIntegerLineEdit("7")
        self.partials_above_entry.setFont(QFont("Arial Nova", 14))
        self.partials_above_entry.set_constraints(0, 128)
        self.partials_above_entry.textChanged.connect(self.update_results)
        self.partials_above_entry.focusIn.connect(self.safe_stop_sounds)
        self.left_layout.addWidget(self.partials_above_label)
        self.left_layout.addWidget(self.partials_above_entry)
        
        self.partials_below_label = QLabel("Below Partials Listed")
        self.partials_below_entry = ReversedDraggableIntegerLineEdit("0")
        self.partials_below_entry.setFont(QFont("Arial Nova", 14))
        self.partials_below_entry.set_constraints(0, 100)
        self.partials_below_entry.textChanged.connect(self.update_results)
        self.partials_below_entry.focusIn.connect(self.safe_stop_sounds)
        self.left_layout.addWidget(self.partials_below_label)
        self.left_layout.addWidget(self.partials_below_entry)
        
        self.edo_label = QLabel("EDO Approximation")
        self.edo_entry = DraggableIntegerLineEdit("12")
        self.edo_entry.setFont(QFont("Arial Nova", 14))
        self.edo_entry.set_constraints(1, 311)
        self.edo_entry.textChanged.connect(self.update_edo_approximation)
        self.edo_entry.focusIn.connect(self.safe_stop_sounds)
        self.left_layout.addWidget(self.edo_label)
        self.left_layout.addWidget(self.edo_entry)

        self.phase_label = QLabel("Phase")
        self.phase_entry = DraggableDecimalLineEdit("0")
        self.phase_entry.min_value = -5.0
        self.phase_entry.max_value = 5.0
        self.phase_entry.setFont(QFont("Arial Nova", 14))
        self.phase_entry.textChanged.connect(self.update_phase_factor)
        self.phase_entry.focusIn.connect(self.safe_stop_sounds)
        self.left_layout.addWidget(self.phase_label)
        self.left_layout.addWidget(self.phase_entry)

        self.roll_off_label = QLabel("Roll-Off")
        self.roll_off_entry = DraggableDecimalLineEdit("1")
        self.roll_off_entry.min_value = -5.0
        self.roll_off_entry.max_value = 5.0
        self.roll_off_entry.setFont(QFont("Arial Nova", 14))
        self.roll_off_entry.textChanged.connect(self.update_roll_off_rate)
        self.roll_off_entry.focusIn.connect(self.safe_stop_sounds)
        self.left_layout.addWidget(self.roll_off_label)
        self.left_layout.addWidget(self.roll_off_entry)

        self.duration_label = QLabel("Playback Duration (s)")
        self.duration_entry = DraggableIntegerLineEdit("2")
        self.duration_entry.setFont(QFont("Arial Nova", 14))
        self.duration_entry.set_constraints(1, 1000)
        self.duration_entry.textChanged.connect(self.update_duration)
        self.duration_entry.focusIn.connect(self.safe_stop_sounds)
        self.left_layout.addWidget(self.duration_label)
        self.left_layout.addWidget(self.duration_entry)

        self.kill_switch_button = QPushButton("Stop Playback")
        self.kill_switch_button.clicked.connect(self.stop_all_sounds)
        self.left_layout.addWidget(self.kill_switch_button)
        
        self.sine_button = QPushButton("Sine Wave")
        self.sine_button.clicked.connect(self.reset_to_sine_wave)
        self.left_layout.addWidget(self.sine_button)

        self.lattice_button = QPushButton("Lattice")
        self.lattice_button.clicked.connect(self.show_lattice_window)
        self.left_layout.addWidget(self.lattice_button)

        self.isohe_button = QPushButton("3HE")
        self.isohe_button.clicked.connect(self.show_isohe_window)
        self.left_layout.addWidget(self.isohe_button)

        self.reset_button = QPushButton("Reset Parameters")
        self.reset_button.clicked.connect(self.reset_parameters)
        self.left_layout.addWidget(self.reset_button)

        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: grey;")
        self.left_layout.addWidget(self.error_label)
        
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_panel.setLayout(self.right_layout)
        self.layout.addWidget(self.right_panel, stretch=9)
        self.header_widget = QWidget()
        self.header_layout = QHBoxLayout()
        self.header_widget.setLayout(self.header_layout)

        self.just_container = QWidget()
        self.just_layout = QVBoxLayout()
        self.just_container.setLayout(self.just_layout)
        self.just_layout.setAlignment(Qt.AlignCenter)

        self.just_intonation_label = QLabel("Just Intonation")
        self.just_intonation_label.setAlignment(Qt.AlignCenter)
        self.just_layout.addWidget(self.just_intonation_label, alignment=Qt.AlignCenter)

        self.series_segment_label = CopyableLabel()
        self.series_segment_label.setAlignment(Qt.AlignCenter)
        self.series_segment_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.series_segment_label.setMaximumWidth(1200)
        self.just_layout.addWidget(self.series_segment_label, alignment=Qt.AlignCenter)

        self.edo_container = QWidget()
        self.edo_layout = QVBoxLayout()
        self.edo_container.setLayout(self.edo_layout)
        self.edo_layout.setAlignment(Qt.AlignCenter)

        self.edo_approximation_label = QLabel("12EDO Approximation")
        self.edo_approximation_label.setAlignment(Qt.AlignCenter)
        self.edo_layout.addWidget(self.edo_approximation_label, alignment=Qt.AlignCenter)

        self.edo_steps_label = CopyableLabel()
        self.edo_steps_label.setAlignment(Qt.AlignCenter)
        self.edo_steps_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.edo_steps_label.setMaximumWidth(1200)
        self.edo_layout.addWidget(self.edo_steps_label, alignment=Qt.AlignCenter)   

        self.header_layout.addWidget(self.just_container, stretch=1)
        self.header_layout.addWidget(self.edo_container, stretch=1)

        self.just_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.edo_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.right_layout.addWidget(self.header_widget)
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Play", "Harmonics", "Tuner Read-Out", "Play", "\\41", "Note Names and Error"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setFont(QFont("Arial Nova", 14))
        self.table.setStyleSheet("""
            QHeaderView::section {
                background-color: rgb(35, 38, 47);
                color: #FFFFFF;
                font-size: 12px;
                padding: 5px;
                border: none;
            }
        """)
        self.right_layout.addWidget(self.table)
        self.waveform_container = QWidget()
        self.waveform_layout = QHBoxLayout()
        self.waveform_container.setLayout(self.waveform_layout)
        self.ji_waveform = WaveformCanvas(is_ji=True)
        self.edo_waveform = WaveformCanvas(is_ji=False)
        self.waveform_layout.addWidget(self.ji_waveform)
        self.waveform_layout.addWidget(self.edo_waveform)
        self.right_layout.addWidget(self.waveform_container)
        self.visualizer = VisualizerWidget(self.isoharmonic_entry, self)
        self.right_layout.addWidget(self.visualizer)
        self.fundamental_entry.textChanged.connect(self.update_results)
        self.isoharmonic_entry.textChanged.connect(self.update_results)
        self.partials_above_entry.textChanged.connect(self.update_results)
        self.partials_below_entry.textChanged.connect(self.update_results)
        self.edo_entry.textChanged.connect(self.update_results)
        self.roll_off_rate = 1.0
        self.phase_factor = 0.0
        self.current_timbre = None
        self.ji_timbre = None
        self.edo_timbre = None
        self.lattice_window = None
        self.isohe_window = None
        self.table.itemChanged.connect(self.trigger_lattice_update)
        
        self.update_results()
        try:
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=2)
            pygame.mixer.set_num_channels(8)
        except Exception as e:
            print(f"Mixer init error: {e}")

    def show_lattice_window(self):
        if not self.lattice_window:
            self.lattice_window = LatticeWindow(self)
        self.lattice_window.show()

    def show_isohe_window(self):
        if not self.isohe_window:
            self.isohe_window = IsoHEWindow(self)
        self.isohe_window.show()

    def trigger_lattice_update(self):
        """Update lattice when parameters change"""
        if self.lattice_window and self.lattice_window.isVisible():
            self.lattice_window.lattice_widget.update_grid() 

    def apply_dark_theme(self):
        dark_stylesheet = """
        QWidget {
            background-color: #23262F;
            color: #FFFFFF;
        }
        QLineEdit {
            background-color: #2C2F3B;
            color: #FFFFFF;
            border: 1px solid #383B47;
            border-radius: 5px;
        }
        QPushButton {
            background-color: #3A3D4A;
            color: #FFFFFF;
            border: none;
            padding: 5px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #4A4D5A;
        }
        QTableWidget {
            background-color: #23262F;
            color: #FFFFFF;
            gridline-color: #383B47;
        }
        QHeaderView::section {
            background-color: #23262F;
            color: #FFFFFF;
            padding: 5px;
            border: none;
        }
        QTableWidgetItem {
            background-color: #23262F;
            color: #FFFFFF;
        }
        QMessageBox {
            background-color: #23262F;
            color: #FFFFFF;
        }
        """
        self.setStyleSheet(dark_stylesheet)

    def generate_combined_waveform(self, frequencies, ratios, duration=0.05, sample_rate=44100):
        t = np.linspace(0, duration, int(sample_rate * duration))
        combined = np.zeros_like(t)
        
        for freq, ratio in zip(frequencies, ratios):
            if self.roll_off_rate != 0:
                effective_ratio = ratio if self.roll_off_rate > 0 else 1/ratio
                amplitude_factor = 1.0 / (float(effective_ratio) ** abs(self.roll_off_rate))
            else:
                amplitude_factor = 1.0
            combined += np.sin(2 * np.pi * freq * t + self.phase_factor) * amplitude_factor
        
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined /= max_val
        
        return combined * normalization_factor

    def reset_parameters(self):
        self.fundamental_entry.setText("1/1")
        self.isoharmonic_entry.setText("1")
        self.partials_above_entry.setText("7")
        self.partials_below_entry.setText("0")
        self.edo_entry.setText("12")
        self.duration_entry.setText("2")
        self.roll_off_entry.setText("1")
        self.phase_entry.setText("0")
        
        self.current_timbre = None
        self.ji_timbre = None
        self.edo_timbre = None
        self.visualizer.current_timbre = None
        self.visualizer.ji_timbre = None
        self.visualizer.edo_timbre = None
        
        self.error_label.setText("Default Parameters")
    
    def reset_to_sine_wave(self):
        try:
            self.current_timbre = None
            self.ji_timbre = None
            self.edo_timbre = None
            
            if hasattr(self, 'visualizer'):
                self.visualizer.current_timbre = None
                self.visualizer.ji_timbre = None
                self.visualizer.edo_timbre = None
            
            self.stop_all_sounds()
            
            self.error_label.setText("Sine wave set")
            
        except Exception as e:
            self.error_label.setText(f"Reset error: {str(e)}")
        
    def update_waveforms(self, iso_series, edo_steps, edo_value):
        try:
            edo = int(edo_value)
            ji_frequencies = [261.6 * float(ratio) for ratio in iso_series]
            ji_signal = self.generate_combined_waveform(ji_frequencies, iso_series)
            
            edo_frequencies = [261.6 * (2 ** (step / edo)) for step in edo_steps]
            edo_signal = self.generate_combined_waveform(edo_frequencies, iso_series)
            
            fundamental_ratio = Fraction(self.fundamental_entry.text())
            freq_fundamental = 261.6 * float(fundamental_ratio)
            vf_signal = self.generate_combined_waveform([freq_fundamental], [fundamental_ratio])
            
            fundamental_cents = ratio_to_cents(float(fundamental_ratio))
            step_edo_str, _ = calculate_edo_step(fundamental_cents, edo)
            step_edo = int(step_edo_str.replace("-", "-"))
            edo_freq_fundamental = 261.6 * (2 ** (step_edo / edo))
            edo_vf_signal = self.generate_combined_waveform([edo_freq_fundamental], [fundamental_ratio])
            
            self.ji_waveform.update_waveform(ji_signal, vf_signal=vf_signal)
            self.edo_waveform.update_waveform(edo_signal, vf_signal=vf_signal, edo_vf_signal=edo_vf_signal)
            self.ji_waveform.draw()
            self.edo_waveform.draw()
        except Exception as e:
            print(f"Error updating waveforms: {e}")

    def update_fundamental_tuner_readout(self):
        try:
            ratio = Fraction(self.fundamental_entry.text())
            cents = ratio_to_cents(float(ratio))
            tuner_readout = assign_12edo_notation(cents)
            self.fundamental_tuner_readout_label.setText(tuner_readout)
            self.error_label.setText("")
        except Exception as e:
            self.fundamental_tuner_readout_label.setText("n/a")
            self.error_label.setText(str(e))

    def update_isoharmonic_tuner_readout(self):
        try:
            ratio = Fraction(self.isoharmonic_entry.text())
            cents = ratio_to_cents(float(ratio))
            tuner_readout = assign_12edo_notation(cents)
            self.isoharmonic_tuner_readout_label.setText(tuner_readout)
            self.error_label.setText("")
        except Exception as e:
            self.isoharmonic_tuner_readout_label.setText("n/a")
            self.error_label.setText(str(e))

    def update_edo_approximation(self):
        try:
            edo = int(self.edo_entry.text())
            self.edo_approximation_label.setText(f"{edo}EDO Approximation")
            self.error_label.setText("")
        except Exception as e:
            self.edo_entry.setText("")
            self.error_label.setText(str(e))

    def update_duration(self):
        try:
            duration = float(self.duration_entry.text())
            if 1 <= duration <= 1000:
                self.visualizer.duration = duration
                self.error_label.setText("")
            else:
                raise ValueError
        except Exception:
            self.duration_entry.setText("")
            self.error_label.setText("1 - 1000s")

    def update_roll_off_rate(self):
        try:
            rate = float(self.roll_off_entry.text())
            if rate < -5.0 or rate > 5.0:
                raise ValueError
            self.roll_off_rate = rate
            self.error_label.setText("")
            self.update_results()
        except ValueError:
            self.roll_off_entry.setText("")
            self.roll_off_rate = 1.0
            self.error_label.setText("-5 to 5")

    def update_phase_factor(self):
        try:
            phase = float(self.phase_entry.text())
            if phase < -5.0 or phase > 5.0:
                raise ValueError
            self.phase_factor = phase
            self.error_label.setText("")
            self.update_results()
        except ValueError:
            self.phase_entry.setText("")
            self.phase_factor = 0.0
            self.error_label.setText("-5 to 5")

    def validate_integer_input(self, input_str, default_value=0):
        return int(input_str) if input_str.strip().isdigit() else default_value

    def update_results(self):
        try:
            fundamental = Fraction(self.fundamental_entry.text())
            isoharmonic = Fraction(self.isoharmonic_entry.text())
            
            max_partials_below = 0
            current_ratio = isoharmonic
            while True:
                current_ratio -= fundamental
                if current_ratio > 0:
                    max_partials_below += 1
                else:
                    break

            safe_max = min(max_partials_below, 128)
            self.partials_below_entry.set_constraints(0, safe_max)
            
            current_val = int(self.partials_below_entry.text() or 0)
            if current_val > safe_max:
                self.partials_below_entry.setText(str(safe_max))
            partials_above = self.validate_integer_input(self.partials_above_entry.text())
            partials_below = self.validate_integer_input(self.partials_below_entry.text())
            edo = int(self.edo_entry.text())
            self.table.horizontalHeaderItem(4).setText(f"\\{edo}")
            series = generate_iso_series(fundamental, isoharmonic, partials_above, partials_below)
            series_segment = format_series_segment(series)
            self.table.setRowCount(0)
            fundamental_cents = ratio_to_cents(float(fundamental))
            step_edo_str, error_edo = calculate_edo_step(fundamental_cents, edo)
            step_edo = int(step_edo_str.replace("-", "-"))
            notation_edo = calculate_single_note(step_edo, edo)
            fundamental_tuner_readout = assign_12edo_notation(fundamental_cents)
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            text_entry_color = QColor(44, 47, 58)
            fundamental_ratio_item = QTableWidgetItem(simplify_ratio(fundamental))
            fundamental_ratio_item.setBackground(text_entry_color)
            fundamental_tuner_readout_item = QTableWidgetItem(fundamental_tuner_readout)
            fundamental_tuner_readout_item.setBackground(text_entry_color)
            self.table.setItem(row_position, 1, fundamental_ratio_item)
            self.table.setItem(row_position, 2, fundamental_tuner_readout_item)
            fundamental_play_button = QPushButton("▶")
            fundamental_play_button.setCheckable(True)
            fundamental_play_button.setStyleSheet("background-color: #2C2F3B; border: none;")
            fundamental_play_button.clicked.connect(lambda _, f=261.6*float(fundamental), r=fundamental, b=fundamental_play_button: self.toggle_play(f, r, b))
            self.table.setCellWidget(row_position, 0, fundamental_play_button)
            edo_freq = 261.6 * (2 ** (step_edo / edo))
            edo_play_button = QPushButton("▶")
            edo_play_button.setCheckable(True)
            edo_play_button.setStyleSheet("background-color: #2C2F3B; border: none;")
            edo_play_button.clicked.connect(lambda _, f=edo_freq, r=fundamental, b=edo_play_button: self.toggle_play(f, r, b))
            self.table.setCellWidget(row_position, 3, edo_play_button)
            step_edo_item = QTableWidgetItem(step_edo_str)
            step_edo_item.setBackground(text_entry_color)
            self.table.setItem(row_position, 4, step_edo_item)
            rounded_error = round(-error_edo)
            error_str = f"{rounded_error:+}".replace("-", "-")
            notation_error_item = QTableWidgetItem(f"{notation_edo}  {error_str}")
            notation_error_item.setBackground(text_entry_color)
            self.table.setItem(row_position, 5, notation_error_item)
            edo_steps = []
            for ratio in series:
                simplified_ratio = simplify_ratio(ratio)
                cents = ratio_to_cents(float(Fraction(simplified_ratio)))
                step_edo_str, error_edo = calculate_edo_step(cents, edo)
                step_edo = int(step_edo_str.replace("-", "-"))
                notation_edo = calculate_single_note(step_edo, edo)
                tuner_readout = assign_12edo_notation(cents)
                row_position = self.table.rowCount()
                self.table.insertRow(row_position)
                play_button = QPushButton("▶")
                play_button.setCheckable(True)
                play_button.setStyleSheet("background-color: #23262F; border: none;")
                play_button.clicked.connect(lambda _, f=261.6*float(Fraction(simplified_ratio)), r=ratio, b=play_button: self.toggle_play(f, r, b))
                self.table.setCellWidget(row_position, 0, play_button)
                self.table.setItem(row_position, 1, QTableWidgetItem(simplified_ratio))
                self.table.setItem(row_position, 2, QTableWidgetItem(tuner_readout))
                edo_freq = 261.6 * (2 ** (step_edo / edo))
                edo_play_button = QPushButton("▶")
                edo_play_button.setCheckable(True)
                edo_play_button.setStyleSheet("background-color: #23262F; border: none;")
                edo_play_button.clicked.connect(lambda _, f=edo_freq, r=ratio, b=edo_play_button: self.toggle_play(f, r, b))
                self.table.setCellWidget(row_position, 3, edo_play_button)
                self.table.setItem(row_position, 4, QTableWidgetItem(step_edo_str))
                rounded_error = round(-error_edo)
                error_str = f"{rounded_error:+}".replace("-", "-")
                self.table.setItem(row_position, 5, QTableWidgetItem(f"{notation_edo}  {error_str}"))
                edo_steps.append(step_edo)
            
            edo_steps_str = "{" + ",".join(map(str, edo_steps)) + "}\\" + str(edo)
            max_header_width = 1200

            series_metrics = self.series_segment_label.fontMetrics()
            elided_series = series_metrics.elidedText(
                series_segment, 
                Qt.ElideNone,
                max_header_width
            )
            self.series_segment_label.setText(elided_series)
            self.series_segment_label.setToolTip(series_segment)

            edo_metrics = self.edo_steps_label.fontMetrics()
            elided_edo = series_metrics.elidedText(
                edo_steps_str, 
                Qt.ElideNone,
                max_header_width
            )
            self.edo_steps_label.setText(elided_edo)
            self.edo_steps_label.setToolTip(edo_steps_str)

            row_position = self.table.rowCount()
            self.table.insertRow(row_position)

            iso_play_all_button = QPushButton("▶")
            iso_play_all_button.setCheckable(True)
            iso_play_all_button.setStyleSheet("background-color: #3A3D4A; border: none;")
            iso_play_all_button.clicked.connect(lambda _, s=series, b=iso_play_all_button: self.toggle_play_series(s, b))
            self.table.setCellWidget(row_position, 0, iso_play_all_button)

            ji_timbre_button = QPushButton("timbre")
            ji_timbre_button.setStyleSheet("background-color: #3A3D4A; border: none; color: white;")
            ji_timbre_button.clicked.connect(self.set_ji_timbre)
            self.table.setCellWidget(row_position, 1, ji_timbre_button)

            ji_save_button = QPushButton(".wav")
            ji_save_button.setStyleSheet("background-color: #3A3D4A; border: none; color: white;")
            ji_save_button.clicked.connect(self.save_ji_wav)
            self.table.setCellWidget(row_position, 2, ji_save_button)

            edo_play_all_button = QPushButton("▶")
            edo_play_all_button.setCheckable(True)
            edo_play_all_button.setStyleSheet("background-color: #3A3D4A; border: none;")
            freqs = [261.6 * (2 ** (step / edo)) for step in edo_steps]
            edo_play_all_button.clicked.connect(lambda _, freqs=freqs, ratios=series, b=edo_play_all_button: self.toggle_play_edo_series(freqs, ratios, b))
            self.table.setCellWidget(row_position, 3, edo_play_all_button)

            edo_timbre_button = QPushButton("timbre")
            edo_timbre_button.setStyleSheet("background-color: #3A3D4A; border: none; color: white;")
            edo_timbre_button.clicked.connect(self.set_edo_timbre)
            self.table.setCellWidget(row_position, 4, edo_timbre_button)

            edo_save_button = QPushButton(".wav")
            edo_save_button.setStyleSheet("background-color: #3A3D4A; border: none; color: white;")
            edo_save_button.clicked.connect(self.save_edo_wav)
            self.table.setCellWidget(row_position, 5, edo_save_button)

            self.visualizer.series = series
            self.visualizer.fundamental = fundamental
            self.visualizer.update()
            self.update_waveforms(series, edo_steps, edo)
            self.trigger_lattice_update()
            if self.isohe_window:
                self.isohe_window.update_equave()
        except Exception as e:
            self.error_label.setText(str(e))

    def set_ji_timbre(self):
        try:
            fundamental = Fraction(self.fundamental_entry.text())
            isoharmonic = Fraction(self.isoharmonic_entry.text())
            partials_above = int(self.partials_above_entry.text())
            partials_below = int(self.partials_below_entry.text())

            series = generate_iso_series(fundamental, isoharmonic, partials_above, partials_below)
            if not series:
                raise ValueError("Empty series")

            lowest = min(series)
            transposed_series = [r / lowest for r in series]

            self.ji_timbre = {
                'ratios': transposed_series,
                'roll_off': self.roll_off_rate,
                'phase': self.phase_factor,
                'base_freq': 261.6
            }
            self.visualizer.ji_timbre = self.ji_timbre
            self.visualizer.current_timbre = self.ji_timbre
            self.error_label.setText("JI timbre set")
        except Exception as e:
            self.error_label.setText(str(e))

    def set_edo_timbre(self):
        try:
            fundamental = Fraction(self.fundamental_entry.text())
            isoharmonic = Fraction(self.isoharmonic_entry.text())
            partials_above = int(self.partials_above_entry.text())
            partials_below = int(self.partials_below_entry.text())
            edo = int(self.edo_entry.text())

            series = generate_iso_series(fundamental, isoharmonic, partials_above, partials_below)
            edo_steps = []

            for ratio in series:
                cents = ratio_to_cents(float(ratio))
                step_str, _ = calculate_edo_step(cents, edo)
                step = int(step_str.replace("-", "-"))
                edo_steps.append(step)

            min_step = min(edo_steps)
            transposed_steps = [s - min_step for s in edo_steps]
            exp_ratios = [2 ** (s / edo) for s in transposed_steps]

            self.edo_timbre = {
                'ratios': exp_ratios,
                'steps': transposed_steps,
                'roll_off': self.roll_off_rate,
                'phase': self.phase_factor,
                'base_freq': 261.6
            }
            self.visualizer.edo_timbre = self.edo_timbre
            self.visualizer.current_timbre = self.edo_timbre
            self.error_label.setText("EDO timbre set")
        except Exception as e:
            self.error_label.setText(str(e))

    def generate_combined_wav_data(self, frequencies, ratios, duration, apply_fade=False): 
        num_samples = int(round(duration * sample_rate))
        t = np.linspace(0, duration, num_samples, False)
        combined = np.zeros(num_samples, dtype=np.float32)
        
        for freq, ratio in zip(frequencies, ratios):
            amplitude_factor = 1.0 / (float(ratio) ** self.roll_off_rate) if self.roll_off_rate != 0 else 1.0
            sine_wave = np.sin(2 * np.pi * freq * t + self.phase_factor)
            
            if apply_fade:
                fade_samples = int(round(fade_duration * sample_rate))
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                sine_wave[:fade_samples] *= fade_in
                sine_wave[-fade_samples:] *= fade_out
            
            combined += sine_wave * normalization_factor * amplitude_factor
        
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined /= max_val
            
        combined_int16 = (combined * max_amplitude).astype(np.int16)
        stereo_buffer = np.zeros((num_samples, 2), dtype=np.int16)
        stereo_buffer[:, 0] = combined_int16
        stereo_buffer[:, 1] = combined_int16
        return stereo_buffer

    def save_wav(self, buffer, filename):
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(buffer.tobytes())
        except Exception as e:
            raise Exception(f"Failed to save WAV file: {e}")

    def sanitize_filename(self, name):
        """Convert musical symbols to ASCII-friendly representations and clean numeric formats"""
        substitutions = {
            '': 'x',       # Double sharp
            '': 'bb',      # Double flat
            '♯': '#',       # Sharp
            '♭': 'b',       # Flat
            '': 'b',       # Custom symbols
            '': '#',
            '': 'd',
            '': 't#',
            '': 't',
            '': 'db',
            '': 'x',
            '': 'bb'
        }
        
        # Convert subscript numbers to normal numbers
        normalized = []
        for c in name:
            if unicodedata.category(c) == 'No':  # Number, other (subscripts)
                normalized.append(str(unicodedata.numeric(c)))
            else:
                normalized.append(substitutions.get(c, c))
        
        # Join and clean decimal zeros
        cleaned = []
        for part in ''.join(normalized).split('_'):
            # Remove trailing .0 from whole numbers
            if '.' in part:
                part = part.rstrip('0').rstrip('.') if '.' in part else part
            cleaned.append(part)
        
        return '_'.join(cleaned)
    
    def save_ji_wav(self):
        try:
            directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save JI WAV Files")
            if not directory:
                return

            rows_to_save = self.table.rowCount() - 1
            for row in range(rows_to_save):
                ratio_item = self.table.item(row, 1)
                tuner_item = self.table.item(row, 2)
                if not (ratio_item and tuner_item):
                    continue
                ratio_str = ratio_item.text()
                tuner_text = tuner_item.text().strip()
                
                # Parse note and cent deviation
                parts = tuner_text.split()
                note_part = parts[0] if parts else ""
                cent_dev = parts[1] if len(parts) > 1 else ""

                try:
                    ratio = Fraction(ratio_str)
                    freq = 261.6 * float(ratio)
                    duration = float(self.duration_entry.text())
                except:
                    continue
                
                if self.ji_timbre:
                    timbre_ratios = self.ji_timbre['ratios']
                    roll_off = self.ji_timbre['roll_off']
                    phase = self.ji_timbre['phase']
                    frequencies = [freq * float(r) for r in timbre_ratios]
                    ratios = timbre_ratios
                else:
                    frequencies = [freq]
                    ratios = [1]
                    roll_off = self.roll_off_rate
                    phase = self.phase_factor
                
                buffer = generate_combined_playback_buffer(
                    frequencies, ratios, duration, roll_off, phase
                )
                
                safe_ratio = self.sanitize_filename(ratio_str.replace('/', '_'))
                safe_note = self.sanitize_filename(note_part)
                
                # Build filename components
                components = [safe_ratio]
                note_part = safe_note
                if cent_dev:
                    note_part += f" {cent_dev}"
                components.append(f"({note_part})")

                filename = " ".join(components) + ".wav"
                filepath = os.path.join(directory, filename)
                self.save_wav(buffer, filepath)
            
            # Combined file generation remains the same
            fundamental = Fraction(self.fundamental_entry.text())
            isoharmonic = Fraction(self.isoharmonic_entry.text())
            partials_above = int(self.partials_above_entry.text())
            partials_below = int(self.partials_below_entry.text())
            series = generate_iso_series(fundamental, isoharmonic, partials_above, partials_below)
            frequencies = [261.6 * float(r) for r in series]
            
            buffer = self.generate_combined_wav_data(frequencies, series, float(self.duration_entry.text()), apply_fade=False)
            
            series_segment = format_series_segment(series)
            combined_filename = f"{series_segment.replace(':', '_').replace('/', '_')}.wav"
            combined_filepath = os.path.join(directory, combined_filename)
            self.save_wav(buffer, combined_filepath)
            
            QMessageBox.information(self, "Success", f"Saved {rows_to_save + 1} WAV files to {directory}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_edo_wav(self):
        try:
            directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save EDO WAV Files")
            if not directory:
                return

            rows_to_save = self.table.rowCount() - 1
            for row in range(rows_to_save):
                step_item = self.table.item(row, 4)
                note_item = self.table.item(row, 5)
                if not (step_item and note_item):
                    continue
                step_str = step_item.text().split('\\')[0].strip()
                note_str = note_item.text().split()[0]
                
                try:
                    step = int(step_str)
                    edo = int(self.edo_entry.text())
                    freq = 261.6 * (2 ** (step / edo))
                    duration = float(self.duration_entry.text())
                except:
                    continue
                
                if self.edo_timbre:
                    timbre_steps = self.edo_timbre['steps']
                    roll_off = self.edo_timbre['roll_off']
                    phase = self.edo_timbre['phase']
                    frequencies = [261.6 * (2 ** ((step + s) / edo)) for s in timbre_steps]
                    ratios = [2 ** (s / edo) for s in timbre_steps]
                else:
                    frequencies = [freq]
                    ratios = [1]
                    roll_off = self.roll_off_rate
                    phase = self.phase_factor
                
                buffer = generate_combined_playback_buffer(
                    frequencies, ratios, duration, roll_off, phase
                )
                
                safe_step = self.sanitize_filename(step_str.replace('\\', '_'))
                safe_note = self.sanitize_filename(note_str.split()[0])
                filename = f"{safe_note} {safe_step}\\{edo}.wav"
                filepath = os.path.join(directory, filename)
                self.save_wav(buffer, filepath)
            
            fundamental = Fraction(self.fundamental_entry.text())
            isoharmonic = Fraction(self.isoharmonic_entry.text())
            partials_above = int(self.partials_above_entry.text())
            partials_below = int(self.partials_below_entry.text())
            edo = int(self.edo_entry.text())
            series = generate_iso_series(fundamental, isoharmonic, partials_above, partials_below)
            steps = []
            frequencies = []
            
            for ratio in series:
                cents = ratio_to_cents(float(ratio))
                step_str, _ = calculate_edo_step(cents, edo)
                step = int(step_str)
                steps.append(step)
                frequencies.append(261.6 * (2 ** (step / edo)))
            
            buffer = self.generate_combined_wav_data(frequencies, series, float(self.duration_entry.text()), apply_fade=False)
            
            steps_str = "{" + ",".join(map(str, steps)) + "}\\" + str(edo)
            combined_filename = f"{steps_str.replace('{', '').replace('}', '').replace('of', '_')}.wav"
            combined_filepath = os.path.join(directory, combined_filename)
            self.save_wav(buffer, combined_filepath)
            
            QMessageBox.information(self, "Success", f"Saved {rows_to_save + 1} WAV files to {directory}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_edo_scl(self):
        try:
            edo = int(self.edo_entry.text())
            steps = []
            for i in range(1, edo + 1):
                cents = (i * 1200.0) / edo
                if i == edo:
                    step = "2/1"
                else:
                    cents_str = f"{cents:.6f}".rstrip('0').rstrip('.')
                    step = cents_str
                comment = f"! {i}\\{edo}"
                steps.append(f" {step} {comment}")
            
            content = [
                f"! {edo}EDO.scl",
                "!",
                f"{edo}EDO",
                f" {edo}",
                "!"
            ] + steps
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Scala File", f"{edo}EDO.scl", "Scala Files (*.scl)"
            )
            if not filename:
                return
            if not filename.endswith('.scl'):
                filename += '.scl'
            
            with open(filename, 'w') as f:
                f.write('\n'.join(content))
            QMessageBox.information(self, "Success", f"Saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_edji_scl(self, equave_ratio):
        try:
            if equave_ratio is None:
                raise ValueError("No valid equave ratio available.")
            edo = int(self.edo_entry.text())
            equave_cents = ratio_to_cents(equave_ratio)
            steps = []
            for i in range(1, edo + 1):
                cents = (i * equave_cents) / edo
                if i == edo:
                    step = f"{equave_ratio.numerator}/{equave_ratio.denominator}"
                else:
                    cents_str = f"{cents:.6f}".rstrip('0').rstrip('.')
                    step = cents_str
                comment = f"! {i}\\{edo}({equave_ratio.numerator}/{equave_ratio.denominator})"
                steps.append(f" {step} {comment}")
            
            ratio_str = f"{equave_ratio.numerator}_{equave_ratio.denominator}"
            default_filename = f"{edo}ED({ratio_str}).scl"
            content = [
                f"! {default_filename}",
                "!",
                f"{edo}ED({equave_ratio.numerator}/{equave_ratio.denominator})",
                f" {edo}",
                "!"
            ] + steps
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Scala File", default_filename, "Scala Files (*.scl)"
            )
            if not filename:
                return
            if not filename.endswith('.scl'):
                filename += '.scl'
            
            with open(filename, 'w') as f:
                f.write('\n'.join(content))
            QMessageBox.information(self, "Success", f"Saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def toggle_play(self, frequency, ratio, button):
        if button is None or not isinstance(button, QPushButton):
            return

        try:
            try:
                roll_off = float(self.roll_off_entry.text())
            except ValueError:
                roll_off = 1.0
                
            roll_off = max(-5.0, min(5.0, roll_off))

            if button.isChecked():
                if (self.visualizer.current_timbre and 
                    not hasattr(self, 'current_timbre')):
                    self.reset_to_sine_wave()
                button.setText("■")
                current_duration = self.visualizer.duration

                if (hasattr(self.visualizer, 'current_timbre') and
                self.visualizer.current_timbre is not None):

                    if (hasattr(self, 'ji_timbre') and 
                    self.visualizer.current_timbre == self.ji_timbre):

                        try:
                            ji_ratios = self.ji_timbre['ratios']
                            frequencies = [frequency * float(r) for r in ji_ratios]
                            
                            buffer = generate_combined_playback_buffer(
                                frequencies,
                                ji_ratios,
                                current_duration,
                                roll_off,
                                self.ji_timbre['phase']
                            )
                            
                            sound = pygame.sndarray.make_sound(buffer)
                            sound.play()
                            self.current_sound = sound

                        except Exception as ji_error:
                            print(f"JI timbre playback error: {ji_error}")
                            button.setChecked(False)
                            button.setText("▶")
                            return

                    elif (hasattr(self, 'edo_timbre') and 
                        self.visualizer.current_timbre == self.edo_timbre):

                        try:
                            edo = int(self.edo_entry.text())
                            base_frequency = frequency
                            exp_ratios = self.edo_timbre['ratios']
                            
                            frequencies = [base_frequency * r for r in exp_ratios]
                            display_ratios = exp_ratios
                            
                            buffer = generate_combined_playback_buffer(
                                frequencies,
                                display_ratios,
                                current_duration,
                                roll_off,
                                self.edo_timbre['phase']
                            )
                            
                            sound = pygame.sndarray.make_sound(buffer)
                            sound.play()
                            self.current_sound = sound

                        except Exception as edo_error:
                            print(f"EDO timbre playback error: {edo_error}")
                            button.setChecked(False)
                            button.setText("▶")
                            return

                else:
                    try:
                        effective_ratio = max(0.0001, float(ratio))
                        if abs(roll_off) > 0.0001:
                            if roll_off < 0:
                                amplitude_factor = effective_ratio ** abs(roll_off)
                            else:
                                amplitude_factor = 1.0 / (effective_ratio ** roll_off)
                        else:
                            amplitude_factor = 1.0
                            
                        self.current_sound = play_sine_wave(
                            frequency, 
                            current_duration, 
                            ratio=ratio,
                            roll_off_rate=roll_off,
                            phase=self.phase_factor
                        )

                    except Exception as sine_error:
                        print(f"Sine wave error: {sine_error}")
                        button.setChecked(False)
                        button.setText("▶")
                        return

                try:
                    QTimer.singleShot(
                        int(current_duration * 1000),
                        lambda: self.safe_reset_button(button)
                    )
                except Exception as timer_error:
                    print(f"Timer error: {timer_error}")
                    self.safe_reset_button(button)

            else:
                button.setText("▶")
                if hasattr(self, 'current_sound') and self.current_sound:
                    try:
                        self.current_sound.stop()
                        self.current_sound = None
                    except Exception as stop_error:
                        print(f"Stop error: {stop_error}")

        except Exception as main_error:
            print(f"Playback system error: {main_error}")
            self.safe_reset_button(button)

    def toggle_play_series(self, series, button):
        if button.isChecked():
            button.setText("■")
            try:
                pygame.mixer.init()
                channel = pygame.mixer.Channel(0)
                
                frequencies = [261.6 * float(r) for r in series]
                buffer = generate_combined_playback_buffer(
                    frequencies,
                    series,
                    self.visualizer.duration,
                    self.roll_off_rate,
                    self.phase_factor
                )
                
                sound = pygame.sndarray.make_sound(buffer)
                channel.play(sound)
                
                self.active_playbacks[button] = {
                    'channel': channel,
                    'timer': QTimer(self)
                }
                
                self.active_playbacks[button]['timer'].singleShot(
                    int(self.visualizer.duration * 1000),
                    lambda: self.reset_play_button(button)
                )
                
            except Exception as e:
                self.error_label.setText(str(e))
                button.setChecked(False)
        else:
            self.stop_playback(button)

    def toggle_play_edo_series(self, freqs, ratios, button):
        if button.isChecked():
            button.setText("■")
            try:
                pygame.mixer.init()
                channel = pygame.mixer.Channel(1)
                
                buffer = generate_combined_playback_buffer(
                    freqs,
                    ratios,
                    self.visualizer.duration,
                    self.roll_off_rate,
                    self.phase_factor
                )
                
                sound = pygame.sndarray.make_sound(buffer)
                channel.play(sound)
                
                self.active_playbacks[button] = {
                    'channel': channel,
                    'timer': QTimer(self)
                }
                
                self.active_playbacks[button]['timer'].singleShot(
                    int(self.visualizer.duration * 1000),
                    lambda: self.reset_play_button(button)
                )
                
            except Exception as e:
                self.error_label.setText(str(e))
                button.setChecked(False)
        else:
            self.stop_playback(button)

    def stop_playback(self, button):
        button.setText("▶")
        if button in self.active_playbacks:
            if self.active_playbacks[button]['channel']:
                self.active_playbacks[button]['channel'].stop()
            
            self.active_playbacks[button]['timer'].stop()
            del self.active_playbacks[button]

    def reset_play_button(self, button):
        if button in self.active_playbacks:
            del self.active_playbacks[button]
        button.setChecked(False)
        button.setText("▶")

    def safe_reset_button(self, button):
        try:
            if button and hasattr(button, 'isChecked'):
                button.setChecked(False)
                button.setText("▶")
        except RuntimeError:
            pass
        except Exception as e:
            print(f"Button reset error: {e}")

    def stop_all_sounds(self):
        try:
            stop_all_sounds()
            
            if hasattr(self, 'current_sound') and self.current_sound:
                stop_sound(self.current_sound)
                self.current_sound = None
                
            for row in range(self.table.rowCount()):
                for col in [0, 3]:
                    button = self.table.cellWidget(row, col)
                    self.safe_reset_button(button)
                    
            if hasattr(self, 'active_button'):
                self.active_button = None
                
        except Exception as e:
            print(f"Stop all error: {e}")
    
    def safe_stop_sounds(self):
        self.stop_all_sounds()
        for timer_info in self.active_playbacks.values():
            if 'timer' in timer_info:
                timer_info['timer'].stop()
        self.active_playbacks.clear()
        
    def closeEvent(self, event):
        self.cleanup_playback()
        event.accept()

    def cleanup_playback(self):
        try:
            self.stop_all_sounds()
            pygame.mixer.quit()
            
            if hasattr(self, 'active_playbacks'):
                for playback in self.active_playbacks.values():
                    if 'timer' in playback and playback['timer'].isActive():
                        playback['timer'].stop()
                self.active_playbacks.clear()
        except Exception as e:
            print(f"Cleanup error: {e}")