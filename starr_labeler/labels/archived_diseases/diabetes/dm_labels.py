
from starr_labeler.labels.extract_labels import *


class dm_labels(labels_base):
    def __init__(self, config, save_name):
        super().__init__(config, save_name)

    def positive_diagnoses(self, merged):
        merged = merged.loc[(merged["ICD10 Code"] >= "E08") & (merged["ICD10 Code"] < "E14")]
        merged = merged[["Patient Id", "Accession Number", "Date", "Imaging_dt"]]
        return merged

    """
    def positive_vitals(self, merged):
        #print(merged)
        merged_sbp = merged.loc[merged['Measure'] == 'SBP']
        merged_dbp = merged.loc[merged['Measure'] == 'DBP']
        high_dbp = merged_dbp.loc[pd.to_numeric(merged_dbp['Value']) >= 80]
        high_sbp = merged_sbp.loc[pd.to_numeric(merged_sbp['Value']) >= 130]
        #pos_dbp = high_dbp.sort_values(['Patient Id', 'Date'])
        #pos_dbp = pos_dbp.groupby(['Patient Id', 'Date']).apply(lambda group: group.iloc[1:, 1:])
        #pos_sbp = high_sbp.sort_values(['Patient Id', 'Date'])
        #print(pos_sbp)
        #pos_sbp = pos_sbp.groupby(['Patient Id', 'Date']).apply(lambda group: group.iloc[1:, 1:])
        #print(pos_sbp)
        merged = pd.concat([high_dbp, high_sbp])
        merged = pd.DataFrame([])

        #merged_egfr = merged.loc[merged['Result'].str.contains('SBP', case = False, regex = True)]
        #merged_egfr = merged_egfr.loc[merged_egfr['Value'] < 60]
        #merged_alb = merged.loc[merged['Result'].str.contains('uacr', case = False, regex = True)]
        #merged_alb = merged_alb.loc[merged_alb['Value'] > 30]
        #print(merged_alb[['Value', 'Result']])
        #merged = merged_egfr[["Patient Id", "Accession Number", "Taken Date", "Imaging_dt"]]
        #merged.columns = ["Patient Id", "Accession Number", "Date", "Imaging_dt"]
        #print(merged)
        return merged
    """
