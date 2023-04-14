import pandas as pd
from pathlib import Path
import sys

from starr_labeler.features.extract_features import extract_base

class extract_demographics(extract_base):
    def __init__(self, config, file_name, feature_type):
        super().__init__(config, file_name, feature_type)

    def process_data(self, pat_data):
        demographics_regex = "|".join(list(self.cfg['FEATURES']['TYPES']['DEMOGRAPHICS']['INCLUDE'].keys()))
        demographics_regex += '|Patient Id'
        pat_data = pat_data.loc[:, list(pd.Series(pat_data.columns).str.contains(demographics_regex, regex = True, case = False, na = False))]
        pat_data.loc[:, 'Date of Birth'] = pd.to_datetime(pat_data.loc[:, 'Date of Birth'], utc=True)
        pat_data.loc[:, 'Race'] = pat_data.loc[:, 'Race'].astype('category').cat.codes
        pat_data.loc[:, 'Gender'] = pat_data.loc[:, 'Gender'].astype('category').cat.codes
        pat_data.loc[:, 'Ethnicity'] = pat_data.loc[:, 'Ethnicity'].astype('category').cat.codes
        return pat_data

    def truncate_data(self, pat_data):
        demographics_regex = "|".join(list(self.cfg['FEATURES']['TYPES']['DEMOGRAPHICS']['INCLUDE'].keys()))
        demographics_regex += '|Patient Id'
        pat_data = pat_data.loc[:, list(pd.Series(pat_data.columns).str.contains(demographics_regex, regex = True, case = False, na = False))]
        truncated = pat_data
        truncated = truncated.sort_values(by=['Patient Id'])
        return truncated



