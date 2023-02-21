import pandas as pd
import os
import sys
import re
from pathlib import Path
from stanford_extract.utils import merge_dfms

from stanford_extract.features.demographics import extract_demographics
from stanford_extract.features.diagnoses import extract_diagnoses
from stanford_extract.features.labs import extract_labs
from stanford_extract.features.vitals import extract_vitals
from stanford_extract.features.procedures import extract_procedures
from stanford_extract.features.med_orders import extract_med_orders
from stanford_extract.features.med_admin import extract_med_admin
from stanford_extract.features.encounters import extract_encounters
from stanford_extract.features.clinical_note_meta import extract_clinical_note_meta
from stanford_extract.features.radiology_report_meta import extract_radiology_report_meta

def process_all_types(cfg, save_truncated):

    if save_truncated:
        cfg_section = cfg['TRUNCATE']
    else:
        cfg_section = cfg['FEATURES']

    features = []
    for feature_type in list(cfg_section['TYPES'].keys()):
        if cfg_section['TYPES'][feature_type]['USE']:
            if (not save_truncated) and (cfg_section['TYPES'][feature_type]['LOAD']):
                print(f"Loading features from {os.path.join(cfg_section['SAVE_DIR'], cfg_section['TYPES'][feature_type]['FILE_NAME'])}.")
                extracted_features = pd.read_csv(os.path.join(cfg_section['SAVE_DIR'], cfg_section['TYPES'][feature_type]['FILE_NAME']))
            else:
                extract_class = getattr(sys.modules[__name__], f"extract_{feature_type.lower()}")
                extract_instance = extract_class(cfg, cfg_section['TYPES'][feature_type]['FILE_NAME'], feature_type, save_truncated)
                if not save_truncated:
                    extracted_features = extract_instance.process_type(fillna = cfg_section['TYPES'][feature_type]['FILL_NA'])
                else:
                    extracted_features = extract_instance.process_type(fillna = 'none')
            if not save_truncated:
                extracted_features["Patient Id"]= extracted_features["Patient Id"].astype(str)
                extracted_features["Accession Number"]= extracted_features["Accession Number"].astype(str)
                features.append(extracted_features)
    if not save_truncated:
        print("Now combining the EHR types into a single input.")
        input_features = merge_dfms(features)
        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        input_features.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in input_features.columns.values]
        input_features.to_csv(os.path.join(cfg['FEATURES']['SAVE_DIR'], 'inputs.csv'), index = False)
        return input_features

def compute_features(cfg):
    return process_all_types(cfg, save_truncated = False)

def truncate(cfg):
    process_all_types(cfg, save_truncated = True)