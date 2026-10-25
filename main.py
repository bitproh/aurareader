"""
main.py - Entry point for AuraReader.
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui_main import ReaderWindow


def main():
    # Enable high DPI scaling
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    app.setApplicationName("AuraReader")
    app.setOrganizationName("AuraReader")

    window = ReaderWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

