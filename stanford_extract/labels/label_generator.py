import os
import pandas as pd
from itertools import chain
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import datetime
from pathlib import Path
from typing import Dict

from stanford_extract.utils import *

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

class label_generator():
    def __init__(self, config, output_folder):
        self.cfg: Dict = config
        self.icd10_codes_regex: str = '|'.join(self.cfg['ICD10']['Codes'])
        self.hierarchical_icd10: bool = self.cfg['ICD10']['Hierarchical']

        if self.cfg['ICD9'] is not None:
            self.icd9_codes_regex: str = '|'.join(self.cfg['ICD9']['Codes'])
            self.hierarchical_icd9: bool = self.cfg['ICD9']['Hierarchical']

        self.output_folder: Path = output_folder
        self.features_path = self.cfg['FEATURES']['PATH']
        self.disease_name = self.cfg['Disease Name']
        # create output folder if it doesn't exist and name it results
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def positive_diagnoses(self, merged):
        if self.hierarchical_icd10:
            pos_icd10 = merged['ICD10 Code'].str.match(self.icd10_codes_regex, case = False, na = False)
        else:
            pos_icd10 = merged['ICD10 Code'].str.fullmatch(self.icd_codes_regex, case = False, na = False)
            
        if self.cfg['ICD9'] is not None:
            if self.hierarchical_icd9:
                pos_icd9 = merged['ICD9 Code'].str.match(self.icd9_codes_regex, case = False, na = False)
            else:
                pos_icd9 = merged['ICD9 Code'].str.fullmatch(self.icd_codes_regex, case = False, na = False)
            merged = merged.loc[pos_icd10 | pos_icd9]
        else:
            merged = merged.loc[pos_icd10]
        merged = merged[["Patient Id", "Accession Number", "Date", "Imaging_dt"]]
        return merged

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
        first_encounters.columns = ['Patient Id', 'First Encounter Date']
        last_encounters = encounters_df.groupby(["Patient Id"], sort = False, as_index = False).apply(lambda x: x.sort_values(['Encounter_dt'], ascending = False).head(1))
        last_encounters.columns = ['Patient Id', 'Last Encounter Date']
        encounters_data = first_encounters.merge(last_encounters, how = 'inner', on = ["Patient Id"])
        return encounters_data

    def compute_diagnosis_dates(self, splits = None, rows = 1000000):
        imaging_df = self.imaging_dates(self.cfg['FEATURES']['DATES'])
        all_mrn_accession = imaging_df[["Patient Id", "Accession Number", "Imaging_dt"]]
        encounters_data = self.encounter_dates()
        mrn_accession = all_mrn_accession.merge(encounters_data, how = 'left', on = ['Patient Id'])

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
                # Rename to Patient Id, Accession Number and Diagnosis Date
                outputs.columns = ["Patient Id", "Accession Number", "Diagnosis Date"]
                types_array.append(outputs)

        final_positive_array = pd.concat(types_array)
        final_positive_array = final_positive_array.drop_duplicates()
            
        results = mrn_accession.merge(final_positive_array, how = 'left', on = ['Patient Id', 'Accession Number'])
        # change name of Imaging_dt to Imaging Date
        results = results.rename(columns = {'Imaging_dt' : 'Imaging Date'})
        print("Saving results...")
        results.to_csv(self.output_folder / (self.disease_name + "_diagnosis_dates.csv"), index = False)
        self.diagnosis_dates = results
        print("Done!")
        print("Total Number of Patients: ", results['Patient Id'].nunique())
        print("Total Number of Images: ", results['Accession Number'].nunique())
        print("Total Number of Positive Patients: ", results.loc[~results['Diagnosis Date'].isna(), 'Patient Id'].nunique())
        
    def compute_diagnosis_labels(self):
        existent_diagnosis_date = ~self.diagnosis_dates['Diagnosis Date'].isna()
        nonexistent_diagnosis_date = self.diagnosis_dates['Diagnosis Date'].isna()
        days_diagnosis_after_imaging = (self.diagnosis_dates['Diagnosis Date'] - self.diagnosis_dates['Imaging Date']).dt.days
        days_last_encounter_after_imaging = (self.diagnosis_dates['Last Encounter Date'] - self.diagnosis_dates['Imaging Date']).dt.days
        diagnosis_date_before_imaging = self.diagnosis_dates['Diagnosis Date'] < self.diagnosis_dates['Imaging Date']

        zeros = (nonexistent_diagnosis_date & (days_last_encounter_after_imaging > 365 * 5)) | (days_diagnosis_after_imaging > 365 * 5)
        ones = existent_diagnosis_date & (days_diagnosis_after_imaging < 365 * 5) & (days_diagnosis_after_imaging > 0)
        twos = existent_diagnosis_date & diagnosis_date_before_imaging
        threes = ~zeros & ~ones & ~twos

        self.diagnosis_dates.loc[zeros, 'Label'] = 0
        self.diagnosis_dates.loc[ones, 'Label'] = 1
        self.diagnosis_dates.loc[twos, 'Label'] = 2
        self.diagnosis_dates.loc[threes, 'Label'] = 3

        self.diagnosis_dates = self.diagnosis_dates[["Patient Id", "Accession Number", "Label"]]

        splits = None
        if splits is not None:
            self.diagnosis_dates = get_splits(splits, self.diagnosis_dates)
            
        self.diagnosis_dates.to_csv(self.output_folder / (self.disease_name + "_diagnosis_labels.csv"), index = False)

        print(f"Number of images in class 0: {np.sum(zeros)}")
        print(f"Number of patients in class 0: {self.diagnosis_dates.loc[self.diagnosis_dates['Label'] == 0]['Patient Id'].nunique()}")
        print(f"Number of images in class 1: {np.sum(ones)}")
        print(f"Number of patients in class 1: {self.diagnosis_dates.loc[self.diagnosis_dates['Label'] == 1]['Patient Id'].nunique()}")
        print(f"Number of images in class 2: {np.sum(twos)}")
        print(f"Number of patients in class 2: {self.diagnosis_dates.loc[self.diagnosis_dates['Label'] == 2]['Patient Id'].nunique()}")
        print(f"Number of images in class 3: {np.sum(threes)}")
        print(f"Number of patients in class 3: {self.diagnosis_dates.loc[self.diagnosis_dates['Label'] == 3]['Patient Id'].nunique()}")

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

    
    