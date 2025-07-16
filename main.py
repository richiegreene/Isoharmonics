import matplotlib
matplotlib.use('Qt5Agg')
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import IsoharmonicApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IsoharmonicApp()
    window.show()
    sys.exit(app.exec_())