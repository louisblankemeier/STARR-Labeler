from pathlib import Path
import sys
from glob import glob
import os

from starr_labeler.labels.extract_labels import *

global code

class ost_labels(labels_base):
    def __init__(self, config, save_name):
        super().__init__(config, save_name)
        
    def positive_diagnoses(self, merged):
        print(f"THIS IS THE CODE {code}")
        all = merged.loc[merged['ICD10 Code'].str.contains("^" + code, na = False)]
        merged = all[["Patient Id", "Accession Number", "Date", "Imaging_dt"]]
        return merged

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    ost_class = ost_labels(cfg, 'labels_ost_frac_vert.csv')

    for code0 in ['S32', 'S52.5', 'S52.6', 'S52.7', 'S52.8', 'S52.9', 'S42.2', 'S30.0', 'S32', 'S32.1', 'S32.3', 'S32.4', 'S32.5', 'S32.7', 'S32.8', 'S72.0', 'S72.1', 'S72.2', 'S72.8', 'S72.9', 'M80', 'M80.0', 'M80.1', 'M80.2', 'M80.3', 'M80.4', 'M80.5', 'M80.8', 'M80.9', 'M48.4', 'M48.5', 'M84.3', 'M84.4']:
        global code
        code = code0
        print(f"Code is {code0}")
        try:
            outputs = ost_class.compute_labels(splits = 5)
        except Exception as e:
            print(e)
            continue