import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class WaveformCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=1, dpi=100, is_ji=True):
        self.fig = Figure(figsize=(width, height), facecolor='none')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_facecolor('none')
        self.ax.axis('off')
        self.is_ji = is_ji
        
        if self.is_ji:
            self.vf_line, = self.ax.plot([], [], color='#0b0656', linewidth=6, zorder=0)
            self.main_line, = self.ax.plot([], [], color='white', linewidth=2, zorder=1)
        else:
            self.edo_vf_line, = self.ax.plot([], [], color='#040224', linewidth=6, zorder=0)
            self.vf_line, = self.ax.plot([], [], color='#0b0656', linewidth=6, zorder=1)
            self.main_line, = self.ax.plot([], [], color='white', linewidth=2, zorder=2)
        
        self.ax.set_xlim(0, 0.4529)
        self.fig.tight_layout(pad=0)

    def update_waveform(self, signal, vf_signal=None, edo_vf_signal=None):
        if len(signal) > 0:
            max_signal = np.max(np.abs(signal))
            if max_signal != 0:
                signal = signal / max_signal
        self.main_line.set_data(np.linspace(0, 1, len(signal)), signal)
        
        if vf_signal is not None:
            if len(vf_signal) > 0:
                max_vf = np.max(np.abs(vf_signal))
                if max_vf != 0:
                    vf_signal = vf_signal / max_vf
            self.vf_line.set_data(np.linspace(0, 1, len(vf_signal)), vf_signal)
        
        if not self.is_ji and edo_vf_signal is not None:
            if len(edo_vf_signal) > 0:
                max_edovf = np.max(np.abs(edo_vf_signal))
                if max_edovf != 0:
                    edo_vf_signal = edo_vf_signal / max_edovf
            self.edo_vf_line.set_data(np.linspace(0, 1, len(edo_vf_signal)), edo_vf_signal)
        
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)
        self.draw()
