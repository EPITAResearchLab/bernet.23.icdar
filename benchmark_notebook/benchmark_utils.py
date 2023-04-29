import sys
import os
from pathlib import Path

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from lsd_path_dataset import *

from typing import List

BIN_FOLDER: Path = os.path.join("..", "bin")
pylene_bin: Path = os.path.join(BIN_FOLDER,  "lsd_pylene")

experiences: List[str] = ["maps", "music_sheets", "trade_directories"]
json_out_filename: Path = Path('lsd_score.json')

# Methods

# Sota
sta_labels: List[str] = [
    "edlines",
    "ocv_hough",
    "cannylines",
    "lsd",
    "lsd_m",
    "elsed",
    "ag3line"
]

# Predictors
predictor_labels: List[str] = [
    'Last observation',
    'SMA',
    'EMA',
    'Double exponential',
    'Kalman',
    'One euro',
]

methods: List[str] = sta_labels + predictor_labels
