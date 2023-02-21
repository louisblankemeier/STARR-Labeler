import os
import pandas as pd
from itertools import chain
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import datetime

from stanford_extract.utils import *
from stanford_extract.features.extract_features import extract_base

pd.options.mode.chained_assignment = None

def get_splits(splits, labels):
    labels = labels.reset_index(drop=True)
    X = labels['Label']
    y = labels['Label']
    groups = labels['Patient Id']
    group_kfold = GroupKFold(n_splits = splits)
    group_kfold.get_n_splits(X, y, groups)
    split_idx = 0
    for _, test_index in group_kfold.split(X, y, groups):
        labels.loc[test_index, 'Split'] = split_idx
        split_idx += 1
    return labels

class labels_base(extract_base):
    def __init__(self, config, save_name):
        self.save_name = save_name
        super().__init__(config, "diagnoses.csv", "diagnoses", save_truncated = False)
        self.cfg = config

    def encounter_dates(self):
        encounters_iterator = data_iterator(self.cfg['FEATURES']['PATH'], 'encounters.csv', None, 10000)

        encounters_dataframes = []
        for encounters in encounters_iterator:
            encounters_dataframes.append(encounters)

        encounters_df = pd.concat(encounters_dataframes)
        encounters_df.loc[:, 'Encounter_dt'] = pd.to_datetime(encounters_df['Date'], format='%m/%d/%Y %H:%M', utc=True)
        encounters_df.loc[encounters_df['Encounter_dt'].dt.date > datetime.date(2022, 2, 1), 'Encounter_dt'] = pd.NaT
        encounters_df = encounters_df[['Patient Id', 'Encounter_dt']]
        first_encounters = encounters_df.groupby(["Patient Id"], sort = False, as_index = False).apply(lambda x: x.sort_values(['Encounter_dt'], ascending = True).head(1))
        first_encounters.columns = ['Patient Id', 'First']
        last_encounters = encounters_df.groupby(["Patient Id"], sort = False, as_index = False).apply(lambda x: x.sort_values(['Encounter_dt'], ascending = False).head(1))
        last_encounters.columns = ['Patient Id', 'Last']
        encounters_data = first_encounters.merge(last_encounters, how = 'inner', on = ["Patient Id"])
        return encounters_data


    def compute_labels(self, splits = None, rows = 1000000):
        imaging_df = self.imaging_dates(self.cfg['FEATURES']['DATES'])
        all_mrn_accession = imaging_df[["Patient Id", "Accession Number", "Imaging_dt"]]
        encounters_data = self.encounter_dates()
        mrn_accession = all_mrn_accession.merge(encounters_data, how = 'left', on = ['Patient Id'])
        #print(all_mrn_accession)
        #print(f"Average duration of encounters: {np.mean(mrn_accession['Last'] - mrn_accession['First'])}")
        #print(f"Standard deviation of duration of encounters: {np.std(mrn_accession['Last'] - mrn_accession['First'])}")

        types_array = []
        
        file_dict = {key : self.cfg['FEATURES']['TYPES'][key]['FILE_NAME'] for key in self.cfg['FEATURES']['TYPES'].keys()}
        for ehr_type in file_dict.keys():
            if hasattr(self, f"positive_{ehr_type.lower()}"):
                #iterator = data_iterator(self.cfg['FEATURES']['PATH'], file_dict[ehr_type], None, rows)
                pat_iter_class = patient_iterator(self.cfg['FEATURES']['PATH'], file_dict[ehr_type], None)
                iterator = iter(pat_iter_class)
                results = []
                print(f"Looking through {ehr_type.lower()}...")
                t = tqdm(total = self.cfg['FEATURES']['NUM_PATIENTS'])
                for pat_data in iterator:
                    num_patients_processed = (pat_data.loc[:, 'Patient Id'].value_counts()).shape[0]
                    merged = pat_data.merge(imaging_df, how = 'left', on = 'Patient Id')
                    positive_function = getattr(self, f"positive_{ehr_type.lower()}")
                    processed = positive_function(merged)
                    processed['Date'] = pd.to_datetime(processed['Date'], format='%m/%d/%Y %H:%M', utc=True)
                    processed = processed.groupby(["Patient Id", "Accession Number"], sort = False, as_index = False).apply(lambda x: x.sort_values(['Date'], ascending = True).head(1))
                    results.append(processed)
                    t.update(num_patients_processed)
                outputs = pd.concat(results)
                outputs = outputs.groupby(["Patient Id", "Accession Number"]).head(1)
                outputs = outputs[["Patient Id", "Accession Number", "Date"]]
                types_array.append(outputs)

        final_positive_array = pd.concat(types_array)
        final_positive_array = final_positive_array.drop_duplicates()
            
        results = mrn_accession.merge(final_positive_array, how = 'left', on = ['Patient Id', 'Accession Number'])
        #results.to_csv("../../checkpoints/intermediate_" + self.save_name, index = False)
        #print(f"Number of patients: {len(list(pd.unique(results.loc[~results['Date'].isna()]['Patient Id'])))}")
        #return

        zeros = (results['Date'].isna() & ((results['Last'] - results['Imaging_dt']).dt.days > 365 * 5)) | ((results['Date'] - results['Imaging_dt']).dt.days > 365 * 5)
        ones = ~results['Date'].isna() & ((results['Date'] - results['Imaging_dt']).dt.days < 365 * 5) & ((results['Date'] - results['Imaging_dt']).dt.days > 0)
        twos = ~results['Date'].isna() & (results['Date'] < results['Imaging_dt']) 
        threes = ~zeros & ~ones & ~twos
        results.loc[zeros, 'Label'] = 0
        results.loc[ones, 'Label'] = 1
        results.loc[twos, 'Label'] = 2
        results.loc[threes, 'Label'] = 3
        results = results[["Patient Id", "Accession Number", "Label"]]
        #print(results)
        if splits is not None:
            results = get_splits(splits, results)
        #print(results)
        results.to_csv(os.path.join(self.cfg['FEATURES']['SAVE_DIR'], self.save_name), index = False)

        print(f"Number of scans in class 0: {np.sum(zeros)}")
        print(f"Number of patients in class 0: {results.loc[results['Label'] == 0]['Patient Id'].value_counts().shape[0]}")
        print(f"Number of scans in class 1: {np.sum(ones)}")
        print(f"Number of patients in class 1: {results.loc[results['Label'] == 1]['Patient Id'].value_counts().shape[0]}")
        print(f"Number of scans in class 2: {np.sum(twos)}")
        print(f"Number of patients in class 2: {results.loc[results['Label'] == 2]['Patient Id'].value_counts().shape[0]}")
        print(f"Number of scans in class 3: {np.sum(threes)}")
        print(f"Number of patients in class 3: {results.loc[results['Label'] == 3]['Patient Id'].value_counts().shape[0]}")