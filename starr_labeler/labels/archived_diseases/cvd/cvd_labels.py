import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import hydra
from ruamel.yaml import YAML

from starr_labeler.labels.extract_labels import *

yaml_dict: Dict = YAML().load(open("config.yaml", "r"))
icd_codes = yaml_dict["ICD10"]
icd_codes_regex = "|".join(icd_codes)


class cvd_labels(labels_base):
    def __init__(self, config, save_name):
        super().__init__(config, save_name)

    def positive_diagnoses(self, merged):
        # merged = merged.loc[merged['ICD10 Code']]
        merged = merged.loc[
            merged["ICD10 Code"].str.fullmatch(icd_codes_regex, case=False, na=False)
        ]
        merged = merged[["Patient Id", "Accession Number", "Date", "Imaging_dt"]]
        return merged


@hydra.main(
    version_base=None,
    config_path="/dataNAS/people/lblankem/opportunistic_ct/libraries/starr_labeler/configs",
    config_name="xgboost.yaml",
)
def main(cfg):
    cvd_class = cvd_labels(cfg, "labels_cvd.csv")
    cvd_class.compute_labels()


main()
