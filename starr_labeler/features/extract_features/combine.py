import pandas as pd
import os
import sys
import re
from pathlib import Path
from starr_labeler.utils import merge_dfms

from starr_labeler.features.ehr_types.demographics import extract_demographics
from starr_labeler.features.ehr_types.diagnoses import extract_diagnoses
from starr_labeler.features.ehr_types.labs import extract_labs
from starr_labeler.features.ehr_types.vitals import extract_vitals
from starr_labeler.features.ehr_types.procedures import extract_procedures
from starr_labeler.features.ehr_types.med_orders import extract_med_orders
from starr_labeler.features.ehr_types.med_admin import extract_med_admin
from starr_labeler.features.ehr_types.encounters import extract_encounters
from starr_labeler.features.ehr_types.clinical_note_meta import extract_clinical_note_meta
from starr_labeler.features.ehr_types.radiology_report_meta import extract_radiology_report_meta

def process_all_types(cfg):
    cfg_section = cfg['FEATURES']

    features = []
    for feature_type in list(cfg_section['TYPES'].keys()):
        if cfg_section['TYPES'][feature_type]['USE']:
            print("")
            print("Now processing feature type: " + feature_type)
            if cfg_section['TYPES'][feature_type]['LOAD']:
                print(f"Loading features from {os.path.join(cfg_section['PATH'], cfg_section['TYPES'][feature_type]['FILE_NAME'])}.")
                extracted_features = pd.read_csv(os.path.join(cfg_section['PATH'], cfg_section['TYPES'][feature_type]['FILE_NAME']))
            else:
                extract_class = getattr(sys.modules[__name__], f"extract_{feature_type.lower()}")
                extract_instance = extract_class(cfg, cfg_section['TYPES'][feature_type]['FILE_NAME'], feature_type)
                extracted_features = extract_instance.process_type(fillna = cfg_section['TYPES'][feature_type]['FILL_NA'])
            extracted_features["Patient Id"]= extracted_features["Patient Id"].astype(str)
            extracted_features["Accession Number"]= extracted_features["Accession Number"].astype(str)
            features.append(extracted_features)
    print("Now combining the EHR types into a single input.")
    input_features = merge_dfms(features)
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    input_features.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in input_features.columns.values]
    input_features.to_csv(os.path.join(cfg['FEATURES']['SAVE_DIR'], 'inputs.csv'), index = False)
    return input_features

def compute_features(cfg):
    return process_all_types(cfg)