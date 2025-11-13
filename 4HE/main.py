import sys
import os
from PyQt5.QtWidgets import QApplication

def main():
    """
    Application entry point.
    Sets up the Python path and launches the main GUI window.
    """
    # --- Critical Path Setup ---
    # The Isoharmonics project has internal packages (like 'theory') that are
    # imported absolutely. To make these imports work when running this new
    # 4HE application, we must add the project's root directory to the path.
    
    # Path to this script: .../Isoharmonics/4HE/main.py
    # Path to 4HE dir: .../Isoharmonics/4HE/
    four_he_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to Isoharmonics dir: .../Isoharmonics/
    isoharmonics_dir = os.path.dirname(four_he_dir)

    # Add the 'Isoharmonics' directory to the path, which contains the 'theory' module.
    sys.path.insert(0, isoharmonics_dir)
    
    # Add the 4HE directory itself to the path to ensure local imports work smoothly
    sys.path.insert(0, four_he_dir)

    # Now that the path is correctly configured, we can import our application
    from gui import FourHEWindow

    app = QApplication(sys.argv)
    window = FourHEWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
