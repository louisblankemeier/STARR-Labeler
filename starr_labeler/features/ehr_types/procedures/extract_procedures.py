############### Adapted from Isabel Gallegos's Imaging Biomarkers Repository. ###############

import pandas as pd
from pathlib import Path
import sys
import os

from starr_labeler.features.extract_features import extract_base

def read_csv_dict(file_name):
    cpt_mapping = pd.read_csv(file_name, header=None)
    cpt_mapping.columns = ['lower', 'upper', 'mapping']
    cpt_mapping.loc[:, 'range'] = cpt_mapping.apply(lambda x : list(range(int(x['lower']), int(x['upper'])+1)), 1)
    cpt_mapping = cpt_mapping[['range', 'mapping']].explode('range')
    cpt_mapping_dict = pd.Series(cpt_mapping.mapping.values,index=cpt_mapping.range).to_dict()
    return cpt_mapping_dict

def map_numeric(procedures_num, file_path):
    cpt_mapping_dict = read_csv_dict(os.path.join(file_path, 'cpt_mapping.csv'))
    procedures_num = procedures_num.assign(mapping = \
        pd.to_numeric(procedures_num.loc[:, 'Code'], 'coerce').map(cpt_mapping_dict))
    procedures_num = procedures_num.dropna(subset=['mapping'])
    return procedures_num

def map_alpha(procedures_alpha, file_path):
    procedures_F =  procedures_alpha[procedures_alpha['Code'].str.contains('[Ff]')]
    procedures_T =  procedures_alpha[procedures_alpha['Code'].str.contains('[Tt]')]
    procedures_other =  procedures_alpha[~procedures_alpha['Code'].str.contains('[FfTt]')]
    procedures_F.loc[:, 'Code'] = procedures_F['Code'].str.strip('F')
    cpt_mapping_F_dict = read_csv_dict(os.path.join(file_path, 'cpt_mapping_F.csv'))
    procedures_F = procedures_F.assign(mapping = \
        pd.to_numeric(procedures_F['Code'], 'coerce').map(cpt_mapping_F_dict))
    procedures_F = procedures_F.dropna(subset=['mapping'])

    procedures_T = procedures_T.assign(mapping="alpha_T")
    procedures_other = procedures_other.assign(mapping="alpha_other")

    procedures_alpha = pd.concat([procedures_F, procedures_T, procedures_other])
    return procedures_alpha

class extract_procedures(extract_base):
    def __init__(self, config, file_name, feature_type):
        super().__init__(config, file_name, feature_type)

    def process_data(self, pat_data):
        pat_data = pat_data.loc[pat_data['Code Type'] == 'CPT']
        pat_data_num = pat_data[~pat_data['Code'].str.contains('[A-Za-z, ,.]')] # louis added period
        pat_data_alpha =  pat_data[pat_data['Code'].str.contains('[A-Za-z, ]')]
        pat_data_num = map_numeric(pat_data_num, self.cfg['FEATURES']['SAVE_DIR'])
        pat_data_alpha = map_alpha(pat_data_alpha, self.cfg['FEATURES']['SAVE_DIR'])
        pat_data = pd.concat([pat_data_num, pat_data_alpha])
        #pat_data.loc[:, 'Date'] = pd.to_datetime(pat_data['Date'], format='%m/%d/%Y %H:%M', utc=True)
        #pat_data.loc[:, 'Date'] = pd.to_datetime(pat_data['Date'], utc=True)
        pat_data.loc[:, 'Value'] = 1
        pat_data = pat_data[['Patient Id', 'mapping', 'Value', 'Date']]
        pat_data.columns = ['Patient Id', 'Type', 'Value', 'Event_dt']
        return pat_data

    def truncate_data(self, pat_data):
        pat_data = pat_data.loc[pat_data['Code Type'] == 'CPT']
        truncated = pat_data[['Patient Id', 'Code Type', 'Code', 'Date']]
        truncated = truncated.sort_values(by=['Patient Id'])
        return truncated
        


