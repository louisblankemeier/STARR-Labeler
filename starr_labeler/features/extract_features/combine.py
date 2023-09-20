import os
import re

import pandas as pd

from starr_labeler.utils.utils import merge_dfms


def process_all_types(cfg):
    cfg_section = cfg

    features = []
    for feature_type in list(cfg_section["EHR_TYPES"].keys()):
        if cfg_section["EHR_TYPES"][feature_type]["USE"]:
            print("Now processing feature type: " + feature_type)
            if cfg_section["EHR_TYPES"][feature_type]["LOAD"]:
                print(
                    f"Loading features from"
                    f"{os.path.join(cfg_section['PATH'], cfg_section['TYPES'][feature_type]['FILE_NAME'])}"
                )
                extracted_features = pd.read_csv(
                    os.path.join(
                        cfg_section["DATA_PATH"],
                        cfg_section["EHR_TYPES"][feature_type]["FILE_NAME"],
                    )
                )
            else:
                module = __import__(
                    f"starr_labeler.features.ehr_types.{feature_type.lower()}",
                    fromlist=[f"{feature_type.lower()}"],
                )
                # split feature type by _ and capitalize each word and then join them
                # back together
                feature_type_camel = "".join(
                    [word.capitalize() for word in feature_type.split("_")]
                )
                extract_class = getattr(module, f"Extract{feature_type_camel}")
                extract_instance = extract_class(
                    cfg,
                    cfg_section["EHR_TYPES"][feature_type]["FILE_NAME"],
                    feature_type,
                )
                extracted_features = extract_instance.process_type(
                    fillna=cfg_section["EHR_TYPES"][feature_type]["FILL_NA"]
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
        for col in input_features.columns
    ]
    input_features.to_csv(os.path.join(cfg["SAVE_DIR"], "features.csv"), index=False)
    print("Done!")
    return input_features


def compute_features(cfg):
    return process_all_types(cfg)
