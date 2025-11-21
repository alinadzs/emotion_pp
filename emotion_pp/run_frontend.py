#!/usr/bin/env python3
"""
Скрипт для запуска Streamlit фронтенда
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # Запуск Streamlit приложения
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "main_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])
