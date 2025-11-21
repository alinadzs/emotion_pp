#!/usr/bin/env python3
"""
Лаунчер проекта Emotion→Meme.
"""

from __future__ import annotations

import subprocess
import sys
import time
import webbrowser
from threading import Thread
from typing import Iterable

REQUIRED = [
    "fastapi",
    "uvicorn",
    "streamlit",
    "requests",
    "numpy",
    "opencv-python",
    "Pillow",
    "deepface",
]

MODULE_ALIASES = {
    "opencv-python": "cv2",
    "Pillow": "PIL",
}


def check_dependencies() -> bool:
    print("🔍 Проверка зависимостей...")
    missing: list[str] = []
    for package in REQUIRED:
        module = MODULE_ALIASES.get(package, package.replace("-", "_"))
        try:
            __import__(module.split(".")[0])
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    if missing:
        print("\n⚠️ Установите отсутствующие пакеты:")
        print("pip install -r requirements.txt")
        return False
    return True


def start_process(command: list[str]) -> subprocess.Popen | None:
    try:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as exc:
        print(f"❌ Не удалось запустить {' '.join(command)}: {exc}")
        return None


def stream_output(process: subprocess.Popen, label: str) -> None:
    if process.stdout is None:
        return
    for line in iter(process.stdout.readline, ""):
        if line:
            print(f"[{label}] {line.rstrip()}")


def main() -> None:
    print("😊 Emotion→Meme launcher")
    print("=" * 40)

    if not check_dependencies():
        return

    api_process = start_process([sys.executable, "run_api.py"])
    if not api_process:
        return

    time.sleep(2)
    frontend_process = start_process([sys.executable, "run_frontend.py"])
    if not frontend_process:
        api_process.terminate()
        return

    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass

    Thread(
        target=stream_output, args=(api_process, "API"), daemon=True
    ).start()
    Thread(
        target=stream_output, args=(frontend_process, "Frontend"), daemon=True
    ).start()

    print("✅ Сервисы запущены. Нажмите Ctrl+C для остановки.")
    try:
        while True:
            time.sleep(1)
            if api_process.poll() is not None or frontend_process.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\n🛑 Остановка...")
    finally:
        for proc in (api_process, frontend_process):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("✅ Сервисы остановлены.")


if __name__ == "__main__":
    main()
