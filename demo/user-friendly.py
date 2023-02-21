import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys
from glob import glob
import os
import wandb
import argparse

path = Path(sys.path[0])
sys.path.insert(0, os.path.join(str(path.parent.absolute()), "Stanford_Extract"))
sys.path.insert(0, os.path.join(str(path.parent.absolute()), "Stanford_Extract/features"))
sys.path.insert(0, str(path.parent.absolute()))

from Stanford_Extract import compute_features, truncate

config_file_path = '../configs/xgboost.yaml'
#truncate(config_file_path)
X = compute_features(config_file_path)