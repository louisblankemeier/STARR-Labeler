import pandas as pd
from pathlib import Path
import sys
import time

from starr_labeler.features.extract_features import extract_base

class extract_vitals(extract_base):
    def __init__(self, config, file_name, feature_type, save_truncated):
        super().__init__(config, file_name, feature_type, save_truncated)

    def process_data(self, pat_data):
        vitals_regex = "|".join(list(self.cfg['FEATURES']['TYPES']['VITALS']['INCLUDE'].keys()))
        pat_data = pat_data[pat_data['Measure'].str.contains(vitals_regex, regex = True, case = False, na = False)]
        pat_data = pat_data[['Patient Id', 'Measure', 'Value', 'Date']]
        pat_data.columns = ['Patient Id', 'Type', 'Value', 'Event_dt']
        
        pat_data.loc[:, 'Value'] = pd.to_numeric(pat_data['Value'], 'coerce')
        pat_data = pat_data[~pd.isna(pat_data.Value)]
        #print(f"Inside process data took {toc - tic:0.4f}")
        return pat_data

    def truncate_data(self, pat_data):
        vitals_regex = "|".join(list(self.cfg['FEATURES']['TYPES']['VITALS']['INCLUDE'].keys()))
        pat_data = pat_data[pat_data['Measure'].str.contains(vitals_regex, regex = True, case = False, na = False)]
        pat_data = pat_data[['Patient Id', 'Measure', 'Value', 'Date']]
        pat_data.loc[:, 'Measure'].loc[pat_data['Measure'] == 'BP'] = 'SBP/DBP'
        to_explode = pat_data.loc[pat_data['Measure'] == 'SBP/DBP']
        to_explode.loc[to_explode['Measure'] == 'SBP/DBP', 'Value'] = pd.DataFrame(to_explode.loc[to_explode['Measure'] == 'SBP/DBP', 'Value'].str.split('/', expand=False))
        to_explode.loc[to_explode['Measure'] == 'SBP/DBP', 'Date'] = pd.DataFrame(to_explode.loc[to_explode['Measure'] == 'SBP/DBP', 'Date'].apply(lambda x: [x, x]))
        to_explode.loc[to_explode['Measure'] == 'SBP/DBP', 'Measure'] = pd.DataFrame(to_explode.loc[to_explode['Measure'] == 'SBP/DBP', 'Measure'].str.split('/', expand=False))
        pat_data = pat_data.loc[pat_data['Measure'] != 'SBP/DBP']
        to_explode = to_explode.set_index(['Patient Id'], append = True).apply(lambda x: x.apply(pd.Series).stack()).reset_index()
        pat_data = pd.concat([pat_data, to_explode])
        truncated = pat_data[['Patient Id', 'Measure', 'Value', 'Date']]
        truncated = truncated.sort_values(by=['Patient Id'])
        return truncated


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    labs_class = extract_vitals(cfg, 'flowsheets.csv')
    input_features = labs_class.compute_FEATURES()
    #print(input_features)


