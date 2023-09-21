# Code for processing STARR electronic health record data
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/louisblankemeier/STARR-Labeler/format.yml?branch=main)

## Installation
```
git clone https://github.com/louisblankemeier/STARR-Labeler
cd STARR-Labeler
conda create --name starr_env python=3.9
conda active starr_env
pip install -e .
```
## Generating Feature Vectors
First, create a config file in starr_labeler/features/configs. You can use the existing config files as templates. Use the following to generate features:
```
cd starr_labeler/features/
python main.py --config-name <name_of_config_file>
```

To run with slurm, edit the command in starr_labeler/features/slurm.py. Then, run:
```
python slurm.py
```

The generated features can be used to train a classifier for a specified disease (label). The output file is called *features.csv*.

### Example features config file snippet:

```
DATA_PATH: /clinical_data/
SAVE_DIR: /STARR-Labeler/starr_labeler/features/results
NUM_PATIENTS: 23547
PREDICTION_DATES: 'CT'
DAYS_AFTER_PREDICTION_DATES: 14
EHR_TYPES:
  DEMOGRAPHICS: 
    USE: True
    FILE_NAME: "demographics.csv"
    USE_COLS: ['Patient Id', 'Gender', 'Race', 'Ethnicity', 'Date of Birth']
    REGEX_TO_FEATURE_NAME: {'Gender': None, 'Race': None, 'Ethnicity': None, 'Date of Birth': None} # Date of Birth here gets converted to age.
    TIME_BINS: 1
    TIME_BIN_DURATION: 1 # in years
    AGGREGATE_ACROSS_TIME: 'mean'
    FILL_NA: 'median'
    SAVE: True
    LOAD: False # whether to load from
  
  LABS: 
    USE: True
    FILE_NAME: "labs.csv"
    USE_COLS: ['Patient Id', 'Value', 'Taken Date', 'Result', 'Reference High', 'Reference Low', 'Units']
    TYPE: 'Result'
    Value: 'Value'
    REGEX_TO_FEATURE_NAME: {'^HDL Cholesterol': 'HDL Cholesterol', 'Cholesterol, Total': 'Cholesterol, Total', 'Creatinine, Ser/Plas': None}
 
```

- A patient can have multiple CT scans over the course of the requested time, but ```NUM_PATIENTS``` corresponds to the number of unique patients.

- If the EHR data was not logged into the system on the same date as the CT Scan, ```DAYS_AFTER_PREDICTION_DATES``` defines how many days after the CT exam the EHR data is still usable/acceptable.

- With ```USE_COLS```, load only specified columns of the csv for improved speed.

- Time resolution (in years) can be increased by getting multiple entries from one subject according to the specified ```TIME_BINS```. If the value is 1, there will be 1 feature vector extracted for the time window specified above. A larger number of bins corresponds to multiple feature vectors extracted from multiple time windows according to the number of bins and the specified bin duration. The number of bins is also shown in the variable name of the output CSV with '_1' for bin 1, '_2' for bin 2 etc.

- If multiple entries are present for one subject specify aggregation strategy with ```AGGREGATE_ACROSS_TIME```.

- The keys in ```REGEX_TO_FEATURE_NAME``` define regex expressions and all values that match the regex are mapped to a single feature with a name given by the value in the dictionary. In the example shown above all variables with names starting from ```HDL Cholesterol``` will be matched.
    

## Generating Outcome Labels
First, create a config file in starr_labeler/labels/disease_configs. You can use the existing config files as templates. Use the following to generate outcome labels:
```
cd starr_labeler/labels/
python main.py --config-name <name_of_config_file>
```

To run with slurm, edit the command in starr_labeler/labels/slurm.py. Then, run:
```
python slurm.py
```

The generated labels can be used to train a classifier for a specified outcome (label) using the EHR features discussed above. The output file is called *labels.csv*.

### Example labels config file snippet:

```
DISEASE_NAME: abdominal_aortic_aneurysm
DAYS_AFTER: 1825
DAYS_BEFORE: 0
CONSIDER_ONLY_FIRST_DIAGNOSIS: True
DATA_PATH: /clinical_data/
SAVE_DIR: /STARR-Labeler/starr_labeler/results
NUM_PATIENTS: 23547
PREDICTION_DATES: 'CT'

EHR_TYPES:
  DIAGNOSES:
    FILE_NAME: "diagnoses.csv"
    USE_COLS: ['Patient Id', 'Date', 'ICD10 Code', 'ICD9 Code']
    ICD10:
      Hierarchical: True
      Codes:
        - I71.0
        - I71.3
        - I71.1
        - I71.4
        - I71.2
    ICD9:
      Hierarchical: True
      Codes:
        - 441.0
        - 441.3
        - 441.1
        - 441.4
```

- ```DAYS_AFTER``` defines the end of the time window in days *after* the CT Scan during which evidence of the disease results in a positive label.
- ```DAYS_BEFORE``` defines the start of the time window in days *before* the CT Scan during which evidence of the disease results in a positive label.
- Class definition:
  - Class 0 &rarr; Patient not diagnosed with the specified disease before or after the CT Scan withing the requested time window or any time in the patient's history.
  - Class 1 &rarr; Patient diagnosed with specified disease during the time window between days before and after.
  - Class 2 &rarr; Patient diagnosed with specified disease earlier than the 'days before' window.
  - Class 3 &rarr; Patient diagnoses with specified disease later than the 'days after' window.
  - Commontly, only patients from Class 0 and 1 are included in the studies as controls/positive classes.
  - In the ```ICD10``` and ```ICD9```, the codes associated with the specified disease need to be defined.


## Citation
Please cite this work if you use it for your research.
