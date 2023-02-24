from pathlib import Path
import sys
from glob import glob
import os
import hydra
from ruamel.yaml import YAML
from typing import List, Dict, Tuple, NamedTuple, Optional

from stanford_extract.labels.generate_labels import generate_labels

@hydra.main(version_base=None, config_path="disease_configs/", config_name = 'cvd.yaml')
def main(cfg):
    output_dir = Path(__file__).parent / "results"
    labels_class = label_generator(cfg, output_dir)
    labels_class.compute_diagnosis_dates()
    labels_class.compute_diagnosis_labels()

main()