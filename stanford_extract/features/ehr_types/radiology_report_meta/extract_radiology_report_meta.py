import pandas as pd
from pathlib import Path
import sys
import numpy as np

from stanford_extract.features.extract_features import extract_base

class extract_radiology_report_meta(extract_base):
    def __init__(self, config, file_name, feature_type, save_truncated):
        super().__init__(config, file_name, feature_type, save_truncated)

    def process_data(self, pat_data):
        #labs_set = set(self.cfg.FEATURES.TYPES.LABS.INCLUDE)
        #pat_data = pat_data.loc[pat_data.loc[:, 'Result'].isin(labs_set)]
        #pat_data.loc[:, 'Value'] = 1
        pat_data = pat_data[['Patient Id', 'Title', 'Age', 'Date']]
        pat_data.columns = ['Patient Id', 'Type', 'Value', 'Event_dt']
        return pat_data

    def truncate_data(self, pat_data):
        return pat_data