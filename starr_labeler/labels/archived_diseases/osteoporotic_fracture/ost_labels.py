from starr_labeler.labels.extract_labels import *


class ost_labels(labels_base):
    def __init__(self, config, save_name):
        super().__init__(config, save_name)

    def positive_diagnoses(self, merged):
        all = merged.loc[merged["ICD10 Code"].str.contains("^M80.08", na=False)]
        merged = all[["Patient Id", "Accession Number", "Date", "Imaging_dt"]]
        return merged
