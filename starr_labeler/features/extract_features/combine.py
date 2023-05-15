import os
import re
import sys

import pandas as pd

from starr_labeler.utils import merge_dfms


def process_all_types(cfg):
    cfg_section = cfg["FEATURES"]

    features = []
    for feature_type in list(cfg_section["TYPES"].keys()):
        if cfg_section["TYPES"][feature_type]["USE"]:
            print("")
            print("Now processing feature type: " + feature_type)
            if cfg_section["TYPES"][feature_type]["LOAD"]:
                print(
                    f"Loading features from"
                    f"{os.path.join(cfg_section['PATH'], cfg_section['TYPES'][feature_type]['FILE_NAME'])}"
                )
                extracted_features = pd.read_csv(
                    os.path.join(
                        cfg_section["PATH"], cfg_section["TYPES"][feature_type]["FILE_NAME"]
                    )
                )
            else:
                extract_class = getattr(sys.modules[__name__], f"extract_{feature_type.lower()}")
                extract_instance = extract_class(
                    cfg, cfg_section["TYPES"][feature_type]["FILE_NAME"], feature_type
                )
                extracted_features = extract_instance.process_type(
                    fillna=cfg_section["TYPES"][feature_type]["FILL_NA"]
                )
            extracted_features["Patient Id"] = extracted_features["Patient Id"].astype(str)
            extracted_features["Accession Number"] = extracted_features["Accession Number"].astype(
                str
            )
            features.append(extracted_features)
    print("Now combining the EHR types into a single input.")
    input_features = merge_dfms(features)
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    input_features.columns = [
        regex.sub("_", col) if any(x in str(col) for x in {"[", "]", "<"}) else col
        for col in set(input_features.columns.values)
    ]
    input_features.to_csv(os.path.join(cfg["FEATURES"]["SAVE_DIR"], "inputs.csv"), index=False)
    return input_features


def compute_features(cfg):
    return process_all_types(cfg)
