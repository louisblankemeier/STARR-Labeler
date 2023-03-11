import os
import pandas as pd
from itertools import chain
from tqdm import tqdm
from pathlib import Path
import sys
from glob import glob
import re
import time

from starr_labeler.utils import *

pd.options.mode.chained_assignment = None

class extract_base:
    def __init__(self, config, file_name, feature_type, save_truncated):
        self.save_truncated = save_truncated
        self.cfg = config
        self.file = file_name
        self.feature_type = feature_type

        if self.save_truncated:
            self.truncate = self.cfg['TRUNCATE']
            self.truncated_path = self.truncate['SAVE_DIR']
            if 'USE_COLS' in self.truncate['TYPES'][self.feature_type.upper()]:
                self.use_cols = self.truncate['TYPES'][self.feature_type.upper()]['USE_COLS']
            else:
                self.use_cols = None
            self.features_path = self.truncate['PATH']
            self.dates = self.truncate['DATES']
            self.num_patients = self.truncate['NUM_PATIENTS']

        else:
            self.features = self.cfg['FEATURES']
            self.features_path = self.features['PATH']
            self.bin_duration = self.features['TYPES'][self.feature_type.upper()]['BIN_DURATION']
            self.bins = self.features['TYPES'][self.feature_type.upper()]['BINS']
            self.lag_after_dates = self.features['LAG_AFTER_DATES']
            self.dates = self.features['DATES']
            self.num_patients = self.features['NUM_PATIENTS']
            self.aggregate = self.features['TYPES'][self.feature_type.upper()]['AGGREGATE']
            if 'USE_COLS' in self.features['TYPES'][self.feature_type.upper()]:
                self.use_cols = self.features['TYPES'][self.feature_type.upper()]['USE_COLS']
            else:
                self.use_cols = None
        
        self.pat_iter_class = patient_iterator(self.features_path, self.file, self.use_cols)
        self.data_path = os.path.join(self.features_path, file_name)
        self.pat_iter = iter(self.pat_iter_class)

    def imaging_dates(self, dates):
        imaging_iterator = data_iterator(self.features_path, 'radiology_report_meta.csv', None, 10000)

        imaging_dataframes = []
        for imaging in imaging_iterator:
            imaging_dataframes.append(imaging.loc[imaging['Type'] == dates])

        imaging_df = pd.concat(imaging_dataframes)
        imaging_df.loc[:, 'Imaging_dt'] = pd.to_datetime(imaging_df['Date'], utc=True)
        imaging_df = imaging_df[['Patient Id', 'Accession Number', 'Imaging_dt']]
        imaging_df = imaging_df.groupby(['Patient Id', 'Accession Number']).head(1)
        cross_walk_data = pd.DataFrame(pd.read_csv(os.path.join(str(Path(self.cfg['FEATURES']['PATH']).parent.parent), 'priority_crosswalk_all.csv'))['accession'])
        cross_walk_data.columns = ['Accession Number']
        cross_walk_data['Accession Number'] = cross_walk_data['Accession Number'].astype(str)
        imaging_df = cross_walk_data.merge(imaging_df, how = 'inner', on = ['Accession Number'])
        return imaging_df

    def pivot(self, merged):
        input_features = merged.loc[(merged.Imaging_dt - merged.Event_dt).dt.days > (-1 * self.lag_after_dates)]

        input_features.loc[:, 'Period'] = '1'
        if self.aggregate == 'pce':
            input_features = input_features.groupby(['Patient Id', 'Accession Number', 'Type', 'Period']).apply(lambda x: pd.DataFrame(x.sort_values(['Event_dt'])['Value']).head(1).mean())
            input_features.columns = ['Value']

        else:
        # Bin temporally 
            for i in range(1, self.bins):
                input_features.loc[(input_features.Imaging_dt - input_features.Event_dt).dt.days > (365 * i * self.bin_duration), 'Period'] = str(i + 1)
            if self.aggregate == 'mean':
                input_features = input_features.groupby(['Patient Id', 'Accession Number', 'Type', 'Period']).mean()
            elif self.aggregate == 'sum':
                input_features = input_features.groupby(['Patient Id', 'Accession Number', 'Type', 'Period']).sum()
            
        input_features = input_features.reset_index()[['Patient Id', 'Accession Number', 'Type', 'Period', 'Value']]
        input_features.loc[:, 'Type'] = input_features['Type'] + '_' + input_features['Period']
        input_features = input_features.drop(columns = 'Period')
        input_features = pd.pivot_table(input_features, 'Value', ['Patient Id', 'Accession Number'], 'Type').reset_index()
        return input_features

    def process_type(self, rows = 1000000, fillna = 'none'):
        save_truncated = self.save_truncated
        if save_truncated:
            print(f"Truncating {self.data_path}...")
        else:
            print(f"Computing features from {self.data_path}...")
        imaging_df = self.imaging_dates(self.dates)
        imaging_df = imaging_df.sort_values(by = ['Patient Id'])
        all_mrn_accession = imaging_df[['Patient Id', 'Accession Number']]

        lab_features = []
        t = tqdm(total = self.num_patients)

        truncated_dfs = []

        for pat_data in self.pat_iter:
            num_patients_processed = len(pat_data['Patient Id'].unique())
            
            if save_truncated:
                truncated = self.truncate_data(pat_data)
                truncated_dfs.append(truncated)
            else:
                pat_data = self.process_data(pat_data)
                merged = pat_data.merge(imaging_df, on = 'Patient Id')

                if 'Type' not in merged.columns:
                    lab_features.append(merged)
                else:
                    merged.loc[:, 'Event_dt'] = pd.to_datetime(merged.loc[:, 'Event_dt'], format='%m/%d/%Y %H:%M', utc=True)
                    merged.loc[:, 'Value'] = pd.to_numeric(merged.loc[:, 'Value'], errors = 'coerce')
                    merged = merged.loc[~pd.isna(merged.Event_dt)]

                    merged = merged[['Patient Id', 'Accession Number', 'Type', 'Value', 'Event_dt', 'Imaging_dt']]
                    input_features = self.pivot(merged)
                    lab_features.append(input_features)
                pat_data = pat_data.reset_index(drop = True)
            t.update(num_patients_processed)

        if not save_truncated:

            lab_input_features = pd.concat(lab_features, axis=0, ignore_index=True)
            lab_input_features = all_mrn_accession.merge(lab_input_features, how = "left", on = ["Patient Id", "Accession Number"])
            if fillna != 'none':
                if fillna == 'median':
                    lab_input_features = lab_input_features.fillna(lab_input_features.median(numeric_only = True))
                else:
                    lab_input_features = lab_input_features.fillna(float(fillna))

            # make the data frame behave well for use with XGBoost
            lab_input_features['Patient Id'] = lab_input_features['Patient Id'].astype("string").str.replace(r'^(0+)', '', regex=True).fillna('0')
            lab_input_features['Accession Number'] = lab_input_features['Accession Number'].astype("string").str.replace(r'^(0+)', '', regex=True).fillna('0')
            #regex = re.compile(r"\[|\]|<", re.IGNORECASE)
            #lab_input_features.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in lab_input_features.columns.values]

            if 'Date of Birth' in lab_input_features:
                lab_input_features.loc[:, 'Age'] = (lab_input_features.loc[:, 'Imaging_dt'] - lab_input_features.loc[:, 'Date of Birth']).dt.days / 365.0
                lab_input_features = lab_input_features.drop(columns = ['Imaging_dt', 'Date of Birth'])

            if self.cfg['FEATURES']['TYPES'][self.feature_type.upper()]['SAVE']:
                if not os.path.isdir(self.cfg['FEATURES']['SAVE_DIR']):
                    os.mkdir(self.cfg['FEATURES']['SAVE_DIR'])
                lab_input_features.to_csv(os.path.join(self.cfg['FEATURES']['SAVE_DIR'], self.file), index = False)
            
            return lab_input_features

        else:
            truncated_all = pd.concat(truncated_dfs)
            truncated_all = truncated_all.sort_values(by = ['Patient Id'])
            if not os.path.isdir(self.truncated_path):
                os.mkdir(self.truncated_path)
            truncated_all.to_csv(os.path.join(self.truncated_path, self.file), index = False)
            return

    def process_data(self):
        pass

    def truncate_data(self):
        pass
