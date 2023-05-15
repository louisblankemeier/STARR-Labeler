import os
import sys
from glob import glob
from pathlib import Path

from starr_labeler.labels.extract_labels import *


class ckd_labels(labels_base):
    def __init__(self, config, save_name):
        super().__init__(config, save_name)

    def positive_diagnoses(self, merged):
        all_ckd = merged.loc[merged["ICD10 Code"].str.contains("^N18", na=False)]
        all_ckd = pd.concat(
            [all_ckd, merged.loc[merged["ICD10 Code"].str.contains("^I12", na=False)]]
        )
        all_ckd = pd.concat(
            [all_ckd, merged.loc[merged["ICD10 Code"].str.contains("^I13", na=False)]]
        )
        all_ckd = pd.concat([all_ckd, merged.loc[merged["ICD10 Code"] == "E08.22"]])
        all_ckd = pd.concat([all_ckd, merged.loc[merged["ICD10 Code"] == "E09.22"]])
        all_ckd = pd.concat([all_ckd, merged.loc[merged["ICD10 Code"] == "E10.22"]])
        all_ckd = pd.concat([all_ckd, merged.loc[merged["ICD10 Code"] == "E11.22"]])
        all_ckd = pd.concat([all_ckd, merged.loc[merged["ICD10 Code"] == "E13.22"]])
        merged = all_ckd[["Patient Id", "Accession Number", "Date", "Imaging_dt"]]
        return merged

    def positive_labs(self, merged):
        merged_egfr = merged.loc[merged["Result"].str.contains("eGFR", case=False, regex=True)]
        merged_egfr = merged_egfr.loc[merged_egfr["Value"] < 60]
        # merged_alb = merged.loc[merged['Result'].str.contains('uacr', case = False, regex = True)]
        # merged_alb = merged_alb.loc[merged_alb['Value'] > 30]
        # print(merged_alb[['Value', 'Result']])
        merged = merged_egfr[["Patient Id", "Accession Number", "Taken Date", "Imaging_dt"]]
        merged.columns = ["Patient Id", "Accession Number", "Date", "Imaging_dt"]
        # print(merged)
        return merged
