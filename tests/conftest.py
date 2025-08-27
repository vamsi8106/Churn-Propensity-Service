# tests/conftest.py
import os, sys
from pathlib import Path

# Add project root (the folder that contains "src/") to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Avoid loading a real model during tests (FastAPI startup hook)
os.environ["SKIP_MODEL_LOAD"] = "1"
