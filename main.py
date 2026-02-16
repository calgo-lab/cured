# Run with: python main.py
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/Home.py"],
        check=True
    )
