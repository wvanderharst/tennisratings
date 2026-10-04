"""All file locations in one place. ROOT = the tennis_model folder; OUT = where fitted parameters and outputs live."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "work")
