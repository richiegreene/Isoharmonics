import sys
import os
from PyQt5.QtWidgets import QApplication

def main():
    """
    Application entry point.
    Sets up the Python path and launches the main GUI window.
    """
    # Now that the path is correctly configured, we can import our application
    from gui import FourHEWindow

    app = QApplication(sys.argv)
    window = FourHEWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
