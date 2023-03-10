import pandas as pd
from pathlib import Path
import sys
import numpy as np

from starr_labeler.features.extract_features import extract_base

class extract_clinical_note_meta(extract_base):
    def __init__(self, config, file_name, feature_type, save_truncated):
        super().__init__(config, file_name, feature_type, save_truncated)

    def process_data(self, pat_data):
        pat_data.loc[:, 'Value'] = 1
        pat_data = pat_data[['Patient Id', 'Type', 'Value', 'Date']]
        pat_data.columns = ['Patient Id', 'Type', 'Value', 'Event_dt']
        return pat_data

    def truncate_data(self, pat_data):
        return pat_data
