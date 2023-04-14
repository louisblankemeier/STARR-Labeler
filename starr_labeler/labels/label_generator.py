import os
import pandas as pd
from itertools import chain
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import datetime
from pathlib import Path
from typing import Dict
import sys

from starr_labeler.utils import *

pd.options.mode.chained_assignment = None

class label_generator():
    """Class for generating labels for the given disease.
    Attributes:
        cfg (Dict): Dictionary containing the configuration parameters.
        icd10_codes_regex (str): Regex for the ICD10 codes.
        hierarchical_icd10 (bool): Whether the ICD10 codes are hierarchical.
        icd9_codes_regex (str): Regex for the ICD9 codes.
        hierarchical_icd9 (bool): Whether the ICD9 codes are hierarchical.
        output_folder (Path): Path to the output folder.
        features_path (Path): Path to the features folder.
        disease_name (str): Name of the disease.
        days_before (int): Number of days before the image date to consider.
        days_after (int): Number of days after the image date to consider.
        diagnosis_dates (pd.DataFrame): Dataframe containing the patient ids, accession numbers, and dates of the images with a positive diagnosis.
    
    Methods:
        positive_diagnoses(merged): Find the positive diagnoses for the given disease.
        encounter_dates(): Find the first encounter date for each patient.
        compute_diagnosis_dates(): Compute the diagnosis dates for the given disease.
        compute_diagnosis_labels(): Compute the diagnosis labels for the given disease.    
    """

    def __init__(self, config):
        """Initialize the label generator.
        Args:
            config (Dict): Dictionary containing the configuration parameters.
            output_folder (Path): Path to the output folder.
        """
        self.cfg: Dict = config
        self.icd10_codes_regex: str = '|'.join(self.cfg['ICD10']['Codes'])
        self.hierarchical_icd10: bool = self.cfg['ICD10']['Hierarchical']

        if 'ICD9' in self.cfg:
            icd9_list = self.cfg['ICD9']['Codes']
            icd9_list = [str(icd9) for icd9 in icd9_list]
            self.icd9_codes_regex: str = '|'.join(icd9_list)
            self.hierarchical_icd9: bool = self.cfg['ICD9']['Hierarchical']

        self.output_folder: Path = Path(__file__).parent / "results" / self.cfg['disease_name']
        self.features_path = self.cfg['FEATURES']['PATH']
        self.disease_name = self.cfg['disease_name']
        self.days_before = self.cfg['days_before']
        self.days_after = self.cfg['days_after']
        self.consider_only_first_diagnosis = self.cfg['consider_only_first_diagnosis']

        # create output folder if it doesn't exist and name it results
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if os.path.exists(self.output_folder / "diagnosis_dates.csv"):
            self.diagnosis_dates = pd.read_csv(self.output_folder / "diagnosis_dates.csv")

    def positive_diagnoses(self, merged):
        """Find the positive diagnoses for the given disease.
        Args:
            merged (pd.DataFrame): Dataframe containing the patient ids, accession numbers, and dates of the images.
        Returns:
            pd.DataFrame: Dataframe containing the patient ids, accession numbers, and dates of the images with a positive diagnosis.
        """

        if self.hierarchical_icd10:
            pos_icd10 = merged['ICD10 Code'].str.match(self.icd10_codes_regex, case = False, na = False)
        else:
            pos_icd10 = merged['ICD10 Code'].str.fullmatch(self.icd10_codes_regex, case = False, na = False)
            
        if 'ICD9' in self.cfg:
            if self.hierarchical_icd9:
                pos_icd9 = merged['ICD9 Code'].str.match(self.icd9_codes_regex, case = False, na = False)
            else:
                pos_icd9 = merged['ICD9 Code'].str.fullmatch(self.icd9_codes_regex, case = False, na = False)
            merged = merged.loc[pos_icd10 | pos_icd9]
        else:
            merged = merged.loc[pos_icd10]
        merged = merged[["Patient Id", "Accession Number", "Date", "Imaging_dt", "Text", "Filename", "ICD10 Code", "ICD9 Code"]]
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
        """Compute the diagnosis dates for the given disease and save them to a csv file.
        Args:
            splits (int, optional): Number of splits to create. Defaults to None.
            rows (int, optional): Number of rows to read at a time. Defaults to 1000000.
        """

        imaging_df = self.imaging_dates(self.cfg['FEATURES']['DATES'])
        all_mrn_accession = imaging_df[["Patient Id", "Accession Number", "Imaging_dt", "Filename", "Text"]]
        encounters_data = self.encounter_dates()
        mrn_accession = all_mrn_accession.merge(encounters_data, how = 'left', on = ['Patient Id'])
        # drop Filename and Text from mrn_accession
        mrn_accession = mrn_accession[["Patient Id", "Accession Number", "Imaging_dt", "First Encounter Date", "Last Encounter Date"]]

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
                    if self.consider_only_first_diagnosis:
                        processed = processed.groupby(["Patient Id", "Accession Number"], sort = False, as_index = False).apply(lambda x: x.sort_values(['Date'], ascending = True).head(1))
                    else:
                        processed = processed.groupby(["Patient Id", "Accession Number"], sort = False, as_index = False).apply(lambda x: x.sort_values(['Date'], ascending = True))
                    results.append(processed)
                    t.update(num_patients_processed)
                outputs = pd.concat(results)
                #outputs = outputs.groupby(["Patient Id", "Accession Number"]).head(1)
                outputs = outputs[["Patient Id", "Accession Number", "Date", "Text", "Filename", "ICD10 Code", "ICD9 Code"]]
                # Rename to Patient Id, Accession Number and Diagnosis Date
                outputs.columns = ["Patient Id", "Accession Number", "Diagnosis Date", "Text", "Filename", "ICD10 Code", "ICD9 Code"]
                types_array.append(outputs)

        final_positive_array = pd.concat(types_array)
        final_positive_array = final_positive_array.drop_duplicates()
            
        results = mrn_accession.merge(final_positive_array, how = 'left', on = ['Patient Id', 'Accession Number'])
        # change name of Imaging_dt to Imaging Date
        results = results.rename(columns = {'Imaging_dt' : 'Imaging Date'})
        print("Saving results...")
        results.to_csv(self.output_folder / "diagnosis_dates.csv", index = False)
        self.diagnosis_dates = results
        print("Done!")
        print("Total Number of Patients: ", results['Patient Id'].nunique())
        print("Total Number of Images: ", results['Accession Number'].nunique())
        print("Total Number of Positive Patients: ", results.loc[~results['Diagnosis Date'].isna(), 'Patient Id'].nunique())


    def compute_diagnosis_labels(self):
        """Compute labels for patients receiving a diagnosis within a time horizon around an imaging exam.
    
        Args:
            days_before (int): Number of days before the imaging exam to consider.
            days_after (int): Number of days after the imaging exam to consider.
        """
        days_before = self.days_before
        days_after = self.days_after

        self.diagnosis_dates['Imaging Date'] = pd.to_datetime(self.diagnosis_dates['Imaging Date'])
        self.diagnosis_dates['Diagnosis Date'] = pd.to_datetime(self.diagnosis_dates['Diagnosis Date'])
        self.diagnosis_dates['Last Encounter Date'] = pd.to_datetime(self.diagnosis_dates['Last Encounter Date'])
        self.diagnosis_dates['First Encounter Date'] = pd.to_datetime(self.diagnosis_dates['First Encounter Date'])

        existent_diagnosis_date = ~self.diagnosis_dates['Diagnosis Date'].isna()
        nonexistent_diagnosis_date = self.diagnosis_dates['Diagnosis Date'].isna()
        days_diagnosis_after_imaging = (self.diagnosis_dates['Diagnosis Date'] - self.diagnosis_dates['Imaging Date']).dt.days
        days_last_encounter_after_imaging = (self.diagnosis_dates['Last Encounter Date'] - self.diagnosis_dates['Imaging Date']).dt.days
        diagnosis_date_before_imaging = (self.diagnosis_dates['Imaging Date'] - self.diagnosis_dates['Diagnosis Date']).dt.days > days_before

        zeros = (nonexistent_diagnosis_date & (days_last_encounter_after_imaging > days_after)) | (days_diagnosis_after_imaging > days_after)
        ones = existent_diagnosis_date & (days_diagnosis_after_imaging < days_after) & (days_diagnosis_after_imaging > (-1 * days_before))
        twos = existent_diagnosis_date & diagnosis_date_before_imaging
        threes = ~zeros & ~ones & ~twos

        self.diagnosis_dates.loc[zeros, 'Label'] = 0
        self.diagnosis_dates.loc[ones, 'Label'] = 1
        self.diagnosis_dates.loc[twos, 'Label'] = 2
        self.diagnosis_dates.loc[threes, 'Label'] = 3

        # save a file with just 1s for label and include Text
        positive_cases = self.diagnosis_dates.loc[self.diagnosis_dates['Label'] == 1]
        positive_cases = positive_cases[["Patient Id", "Accession Number", "Filename", "Imaging Date", "Diagnosis Date", "ICD10 Code", "ICD9 Code", "Text"]]
        print(positive_cases.columns)
        positive_cases.to_csv(self.output_folder / "positive_cases.csv", index = False)


        self.diagnosis_dates = self.diagnosis_dates[["Patient Id", "Accession Number", "Label"]]

        splits = None
        if splits is not None:
            self.diagnosis_dates = get_splits(splits, self.diagnosis_dates)
            
        self.diagnosis_dates.to_csv(self.output_folder / "diagnosis_labels.csv", index = False)

        print(f"Number of images in class 0 (no diagnosis before imaging date + days_after and sufficient followup): {np.sum(zeros)}")
        print(f"Number of patients in class 0 (no diagnosis before imaging date + days_after and sufficient followup): {self.diagnosis_dates.loc[self.diagnosis_dates['Label'] == 0]['Patient Id'].nunique()}")
        print(f"Number of images in class 1 (diagnosis between imaging date - days_before and imaging date + days_after): {np.sum(ones)}")
        print(f"Number of patients in class 1 (diagnosis between imaging date - days_before and imaging date + days_after): {self.diagnosis_dates.loc[self.diagnosis_dates['Label'] == 1]['Patient Id'].nunique()}")
        print(f"Number of images in class 2 (diagnosis before imaging date - days_before): {np.sum(twos)}")
        print(f"Number of patients in class 2 (diagnosis before imaging date - days_before): {self.diagnosis_dates.loc[self.diagnosis_dates['Label'] == 2]['Patient Id'].nunique()}")
        print(f"Number of images in class 3 (other): {np.sum(threes)}")
        print(f"Number of patients in class 3 (other): {self.diagnosis_dates.loc[self.diagnosis_dates['Label'] == 3]['Patient Id'].nunique()}")

    def imaging_dates(self, dates):
        """Get imaging dates.
        """

        radiology_report_iterator = data_iterator(self.features_path, 'radiology_report.csv', None, 10000)
        radiology_report_dataframes = []
        for radiology_report in radiology_report_iterator:
            radiology_report_dataframes.append(radiology_report)
        radiology_report_df = pd.concat(radiology_report_dataframes)
        radiology_report_df = radiology_report_df[['Accession Number', 'Text']]

        imaging_iterator = data_iterator(self.features_path, 'radiology_report_meta.csv', None, 10000)

        imaging_dataframes = []
        for imaging in imaging_iterator:
            imaging_dataframes.append(imaging.loc[imaging['Type'] == dates])

        imaging_df = pd.concat(imaging_dataframes)
        imaging_df.loc[:, 'Imaging_dt'] = pd.to_datetime(imaging_df['Date'], utc=True)
        imaging_df = imaging_df[['Patient Id', 'Accession Number', 'Imaging_dt']]
        imaging_df = imaging_df.groupby(['Patient Id', 'Accession Number']).head(1)
        cross_walk_data = pd.DataFrame(pd.read_csv(os.path.join(str(Path(self.cfg['FEATURES']['PATH']).parent), 'priority_crosswalk_all.csv')))
        cross_walk_data = cross_walk_data[['accession', 'filename']]
        cross_walk_data.columns = ['Accession Number', 'Filename']
        cross_walk_data['Accession Number'] = cross_walk_data['Accession Number'].astype(str)
        cross_walk_data['Filename'] = cross_walk_data['Filename'].astype(str)
        imaging_df = cross_walk_data.merge(imaging_df, how = 'inner', on = ['Accession Number'])

        imaging_df = imaging_df.merge(radiology_report_df, how = 'inner', on = ['Accession Number'])
        return imaging_df

    
    