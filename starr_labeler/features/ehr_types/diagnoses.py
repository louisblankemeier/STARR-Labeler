import pandas as pd

from starr_labeler.features.extract_features.extract import extract_base


class extract_diagnoses(extract_base):
    def __init__(self, config, file_name, feature_type):
        super().__init__(config, file_name, feature_type)

    def process_data(self, pat_data):
        diagnoses_type = self.cfg["EHR_TYPES"]["DIAGNOSES"]["TYPE"]
        pat_data.loc[:, diagnoses_type] = pat_data.loc[:, diagnoses_type].str.split(",")
        pat_data = pat_data.explode(diagnoses_type).reset_index(drop=True)

        if "REGEX_TO_FEATURE_NAME" in self.cfg["EHR_TYPES"]["DIAGNOSES"]:
            diagnoses_regex = "|".join(
                list(self.cfg["EHR_TYPES"]["DIAGNOSES"]["REGEX_TO_FEATURE_NAME"].keys())
            )
            pat_data = pat_data.loc[
                pat_data[diagnoses_type].str.match(
                    diagnoses_regex, case=False, na=False
                )
            ]

        pat_data = pat_data.loc[~pd.isna(pat_data.loc[:, diagnoses_type])]
        pat_data.loc[:, diagnoses_type] = pat_data.loc[:, diagnoses_type].map(
            lambda x: "".join(x.split(".", 1)).strip()[
                0 : self.cfg["EHR_TYPES"]["DIAGNOSES"]["NUM_ICD_CHARS"]
            ]
        )
        pat_data = pat_data[["Patient Id", diagnoses_type, "Date"]]

        if not pat_data.empty:
            pat_data.loc[:, "Value"] = 1
        else:
            pat_data = pd.DataFrame(
                columns=["Patient Id", diagnoses_type, "Date", "Value"]
            )
        pat_data = pat_data[["Patient Id", diagnoses_type, "Value", "Date"]]
        pat_data.columns = ["Patient Id", "Type", "Value", "Event_dt"]
        return pat_data

    def truncate_data(self, pat_data):
        truncated = pat_data[["Patient Id", "Date", "ICD10 Code"]]
        truncated = truncated.sort_values(by=["Patient Id"])
        return truncated
