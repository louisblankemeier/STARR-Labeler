import pandas as pd
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import inspect
from pathlib import Path
import sys

path = Path(sys.path[0])
sys.path.insert(0, str(path.parent.absolute()))

from utils import data_iterator, frequency_by_num_patients, setup_cfg, get_parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    iterator = data_iterator(os.path.join(cfg.FEATURES.PATH, 'clinical_data'), 'labs.csv')
    frequency_by_num_patients(iterator, 'Lab', './lab_broader_frequencies_by_patient.png', 50, 20)