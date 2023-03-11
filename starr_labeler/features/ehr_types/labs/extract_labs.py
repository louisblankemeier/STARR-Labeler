import pandas as pd
from pathlib import Path
import sys
import numpy as np

from stanford_extract.features.extract_features import extract_base

class extract_labs(extract_base):
    def __init__(self, config, file_name, feature_type, save_truncated):
        super().__init__(config, file_name, feature_type, save_truncated)

    def process_data(self, pat_data):
        labs_regex = "|".join(list(self.cfg['FEATURES']['TYPES']['LABS']['INCLUDE'].keys()))

        if 'Hemoglobin A1c' in labs_regex:
            pat_data = pat_data.replace('Hb A1c Diabetic Assessment', 'Hemoglobin A1c')

        pat_data = pat_data.loc[pat_data['Result'].str.contains(labs_regex, regex = True, case = False, na = False)]

        for key, value in self.cfg['FEATURES']['TYPES']['LABS']['INCLUDE'].items():
            if value != "None":
                pat_data.loc[pat_data['Result'].str.contains(key, regex = True, case = False, na = False), 'Result'] = value

        pat_data.loc[:, 'Value'] = pd.to_numeric(pat_data.loc[:, 'Value'], 'coerce')
        pat_data.loc[:, 'Reference High'] = pd.to_numeric(pat_data.loc[:, 'Reference High'], 'coerce')
        pat_data.loc[:, 'Reference Low'] = pd.to_numeric(pat_data.loc[:, 'Reference Low'], 'coerce')
        high_low_nan = np.logical_and(pd.isna(pat_data.loc[:, 'Reference High']), pd.isna(pat_data.loc[:, 'Reference Low']))
        pat_data.loc[high_low_nan, 'norm'] = pat_data.loc[:, 'Value']
        pat_data.loc[~high_low_nan, 'norm'] = (pat_data.loc[~high_low_nan, 'Value'] - pat_data.loc[~high_low_nan, 'Reference Low']) / (pat_data.loc[~high_low_nan, 'Reference High'] - pat_data.loc[~high_low_nan, 'Reference Low'])
        pat_data = pat_data[['Patient Id', 'Result', 'Value', 'Taken Date']]
        pat_data.columns = ['Patient Id', 'Type', 'Value', 'Event_dt']
        return pat_data

    def truncate_data(self, pat_data):
        labs_regex = "|".join(list(self.cfg['FEATURES']['TYPES']['LABS']['INCLUDE'].keys()))

        if 'Hemoglobin A1c' in labs_regex:
            pat_data = pat_data.replace('Hb A1c Diabetic Assessment', 'Hemoglobin A1c')

        pat_data = pat_data.loc[pat_data['Result'].str.contains(labs_regex, regex = True, case = False, na = False)]

        for key, value in self.cfg['FEATURES']['TYPES']['LABS']['INCLUDE'].items():
            if value != "None":
                pat_data.loc[pat_data['Result'].str.contains(key, regex = True, case = False, na = False), 'Result'] = value

        truncated = pat_data[['Patient Id', 'Value', 'Taken Date', 'Result', 'Reference High', 'Reference Low', 'Units']]
        truncated.loc[truncated['Result'] == 'eGFR Refit Without Race (2021)', 'Reference Low'] = 0
        truncated.loc[truncated['Result'] == 'eGFR Refit Without Race (2021)', 'Reference High'] = 60
        repl = lambda m: str((int(m.group(1)) + int(m.group(2))) / 2)
        truncated['Value'] = truncated['Value'].str.replace(r' *(\d) *- *(\d) *', repl, regex=True)
        truncated['Reference High'] = truncated['Reference High'].astype(str).str.strip('<>= ')
        truncated['Reference Low'] = truncated['Reference Low'].astype(str).str.strip('<>= ')
        truncated['Value'] = truncated['Value'].astype(str).str.strip('<>= ')
        truncated['Value'] = pd.to_numeric(truncated['Value'], errors = 'coerce')
        truncated['Reference High'] = pd.to_numeric(truncated['Reference High'], errors = 'coerce')
        truncated['Reference Low'] = pd.to_numeric(truncated['Reference Low'], errors = 'coerce')
        truncated = truncated.sort_values(by=['Patient Id'])
        return truncated
