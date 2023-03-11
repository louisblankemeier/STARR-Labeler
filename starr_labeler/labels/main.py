from pathlib import Path
import sys
from glob import glob
import os
import hydra
from ruamel.yaml import YAML
from typing import List, Dict, Tuple, NamedTuple, Optional

from stanford_extract.labels.label_generator import label_generator
from stanford_extract.labels.mcm_generator import mcm_generator

@hydra.main(version_base=None, config_path="disease_configs/", config_name = 'cvd.yaml')
def main(cfg):
    '''
    labels_class = label_generator(cfg)
    labels_class.compute_diagnosis_dates()
    labels_class.compute_diagnosis_labels()
    '''
    mcm_class = mcm_generator(cfg)
    mcm_class.generate_mcm_json()
    mcm_class.generate_mcm_dataframe()
    mcm_class.run_hyde()
main()