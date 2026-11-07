import sys
from pathlib import Path

# Make the task module importable when pytest runs from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
