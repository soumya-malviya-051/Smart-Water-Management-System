import os
import sys

# Ensure src/ is importable
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Run the streamlit application
if __name__ == "__main__":
    os.system(f"streamlit run {os.path.join(SRC_DIR, 'dashboard.py')}")
