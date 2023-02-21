# Code for processing EHR data from Stanford Hospital
Much of the code in this repo is adapted from Isabel Gallegos's Imaging Biomarkers Repository.

## Usage

```
import hydra
from stanford_extract.labels.ischemic_heart_disease import ihd_labels

@hydra.main(version_base=None, config_path="./configs", config_name="xgboost")
def compute_labels(cfg):
    ihd_class = ihd_labels(cfg, 'labels_ihd.csv')
    outputs = ihd_class.compute_labels(splits = 5)

compute_labels()
```

## Features Overview
- Configurable EHR processing pipeline. 
- Configure types of EHR to include. Current support exists for demographics, vitals, labs, diagnoses, procedures, medication orders, medication admin, encounters, radiology reports meta data, clinical notes meta data.
- Base class implementation shared by all EHR types minimizes the amount of EHR type specific code. New EHR types can be included with just a few lines of code. Although, some EHR types require more custom processing code than others.
- Configure bining strategy - number of bins, bin duration, and aggregation function. Currently support sum, mean, and pce where pce takes the mean of the most recent 3 measurements.
- Configure imputation strategy. Currently, median imputation and constants are supported.
- Configure features to include from EHR using regex functionality. 
- Function for combining EHR types to generate a feature vector for each patient.
- Base class can process large amount of data using a custom iterable.
- Configure caching and loading truncated versions of the EHR during various intermediate processing steps. These cached files can be accessed during future processing iterations, making EHR processing faster and facilitating quicker experimentation. Time spent on profiling code and improving performance. 
- Functionality for visualizing EHR data, including patient trajectories.
- Functionality for labeling patients with various diseases. Current support for IHD, HTN, OSTEO, DM, and CKD. 
- Labeling base class allows for implementing labels for new diseases using just a few lines of code. 
- Functionality for generating splits for cross-validation. 
- Configurable diagnosis criteria that determines patient labeling. 
- Code that demonstrates using the feature generation and labeling functionality for future disease incidence prediction, using XGBoost, pooled cohort equations, fully-connected network, and SAINT.
- Clean API with weights and biases support, informative outputs, and progress bars. 

## Extracting feature from a single EHR modality

To create features from any type of EHR data, all you have to do is override the process_data function in the extract_base class. An example of this is shown at the bottom of the section. In this function, you simply need to return a dataframe that has one of two formats. Either the DataFrame should contain the following column names: 'Patient Id', 'Type', 'Value', and 'Event_dt', or the DataFrame should contain one row per patient where the columns are to be merged directly into the patient's feature representation. The base_class will detect the format of the DataFrame automatically based on the presence of the 'Type' column. When one calls the function compute_features on the class, the following steps occur:

1) An iterable is generated for the file of interest, given by the file_name __init__ argument. This iterable tries to extract a specified number of rows at a time from the file, but also ensures that for a single patient, patient data is not split across calls to the iterable's next() function. This ensures that subsequent processing steps aggregate all data for a single patient in the correct way.
2) A DataFrame containing a list of dates of interest for each patient is generated, such that for each patient, EHR is only included from before these dates. This is useful for prediction tasks where EHR from later dates can provide labels. 
3) The overridden function, process_data, is called which extracts data using a next call to the iterable.
4) If there is no 'Type' column name in the DataFrame, the DataFrame containing the list of dates of interest for each patient is merged with the DataFrame returned by process_data. Steps 3 and 4 continue until all of the EHR data is processed. 
5) If there is a 'Type' field, the following steps occur:
6) The 'Value' field of the returned DataFrame is transformed into a numerical data type, and the 'Event_dt' field is transformed into datetime objects. 
7) The DataFrame containing the list of dates of interest for each patient is merged with the DataFrame returned by process_data.
8) A column is then added to the resulting DataFrame indicating the period of time relative to the imaging date. Parameters controlling period membership are given by the FEATURES.TYPE.BIN and FEATURES.TYPE.BIN_DURACTION parameters in the config file. 
9) The resulting DataFrame is then grouped by 'Patient Id', 'Accession Number', 'Type', and 'Period', and an aggregation function specified by FEATURES.TYPE.AGGREGATE is applied.
10) The DataFrame is then pivoted such that the values from the 'Type' column become column headers. The resulting DataFrame should have one feature vector per accession number. Steps 3, 6, 7, 8, 9, and 10 occur until all of the EHR data is processed. 

```
from extract_features import extract_base

class extract_med_orders(extract_base):
    def __init__(self, config, file_name, feature_type):
        super().__init__(config, file_name, feature_type)

    def process_data(self, pat_data, return_truncated):
        pat_data.loc[:, 'Value'] = 1
        pat_data = pat_data[['Patient Id', 'Medication', 'Value', 'Start Date']]
        pat_data.columns = ['Patient Id', 'Type', 'Value', 'Event_dt']
        return pat_data
```


## Combining multiple EHR modalities

Calling generate_inputs will run the steps in the above section for each type of EHR data where USE = True in the configuration file. This function automatically searches and finds a corresponding class for each EHR modalities. If a class has not been defined for an EHR modality, but USE = True in the configuration file, and error will be thrown. It will then merge all the types of EHR data into a single feature vector for each accession number. 

## Example figures

<img src="https://github.com/louisblankemeier/Stanford_Extract/blob/main/figures/ehr_types_ablation_study1.png" width="1000">

<img src="https://github.com/louisblankemeier/Stanford_Extract/blob/main/figures/ehr_types_ablation_study2.png" width="1000">

<img src="https://github.com/louisblankemeier/Stanford_Extract/blob/main/figures/lab_frequencies_by_patient.png" width="1000">

<img src="https://github.com/louisblankemeier/Stanford_Extract/blob/main/figures/trajectories_pat_1.png" width="1000">





