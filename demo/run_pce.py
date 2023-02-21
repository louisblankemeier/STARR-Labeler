import xgboost as xgb
import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys
from glob import glob
import os
import wandb
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy.stats import bootstrap

from pce import *

path = Path(sys.path[0])
sys.path.insert(0, os.path.join(str(path.parent.absolute()), "Stanford_Extract"))
sys.path.insert(0, os.path.join(str(path.parent.absolute()), "Stanford_Extract/features"))
sys.path.insert(0, str(path.parent.absolute()))

from Stanford_Extract import compute_features, get_parser, setup_cfg


config_file_path = '../configs/pce.yaml'

args = get_parser().parse_args()
args.config_file = config_file_path
cfg = setup_cfg(args)

wandb.login(key = 'b99f216f5cf64b99d55527e8c307c9858063cbbe')
wandb.init(project = "feb_22_cohort_ehr_processing", group = "pce", config = cfg)

num_splits = 5


#features = compute_features(config_file_path, save_path = '../Stanford_Extract/checkpoints/inputs_pce_v1.csv')


features = pd.read_csv('../Stanford_Extract/checkpoints/inputs_pce_v1.csv')

y = pd.read_csv('../Stanford_Extract/checkpoints/labels.csv')

data = features.merge(y, how = 'inner', on = ['Patient Id', 'Accession Number'])
zero_or_one = np.logical_or(data.iloc[:, -2] == 0, data.iloc[:, -2] == 1)
data = data.loc[zero_or_one, :]
features = data.iloc[:, 2:-2]
y = data.iloc[:, -2]
y = np.array(y)
splits = data.iloc[:, -1]
splits = np.array(splits)
imaging_dates = data.loc[:, 'Imaging_dt']
pt_ids = data.loc[:, 'Patient Id']

#mean_bf = np.zeros([features.shape[0], 5])
#mean_wf = np.zeros([features.shape[0], 5])
#mean_bm = np.zeros([features.shape[0], 5])
#mean_wm = np.zeros([features.shape[0], 5])

mean_bf = []
mean_wf = []
mean_bm = []
mean_wm = []

scores = []

train_data = features.loc[splits < 4]

for i in range(train_data.shape[0]):
    row = train_data.iloc[i]
    #print(row)
    if row.loc['Gender'] == 0:
        gender = 'F' 
    else:
        gender = 'M'

    if row.loc['Race'] == 1:
        race = 'B' 
    else:
        race = 'W'

    diabetes = row.loc['E08_1'] or row.loc['E09_1'] or row.loc['E10_1'] or row.loc['E11_1'] or row.loc['E13_1']
    smoking = row.loc['F17_1'] or row.loc['Z72_1']

    X = [gender, row.loc['Age'], row.loc['Cholesterol, Total_1'], row.loc['HDL Cholesterol_1'], row.loc['SBP_1'], smoking, diabetes, bool(row.loc['Blood Pressure_1']), race, 10]
    score = frs(*X)

    if gender == 'F':
        if race == "B":
            mean_bf.append(score)
        if race == "W":
            mean_wf.append(score)

    if gender == 'M':
        if race == "B":
            mean_bm.append(score)
        if race == "W":
            mean_wm.append(score)

bf = np.mean(mean_bf)
print(f"bf: {bf}")
wf = np.mean(mean_wf)
print(f"wf: {wf}")
bm = np.mean(mean_bm)
print(f"bm: {bm}")
wm = np.mean(mean_wm)
print(f"wm: {wm}")

risks = []

test_data = features.loc[splits == 4]

for i in range(test_data.shape[0]):
    row = test_data.iloc[i]
    #print(row)
    if row.loc['Gender'] == 0:
        gender = 'F' 
    else:
        gender = 'M'

    if row.loc['Race'] == 1:
        race = 'B' 
    else:
        race = 'W'

    if gender == 'F':
        if race == "B":
            mean = bf
        if race == "W":
            mean = wf

    if gender == 'M':
        if race == "B":
            mean = bm
        if race == "W":
            mean = wm

    diabetes = row.loc['E08_1'] or row.loc['E09_1'] or row.loc['E10_1'] or row.loc['E11_1'] or row.loc['E13_1']
    smoking = row.loc['F17_1'] or row.loc['Z72_1']

    X = [gender, row.loc['Age'], row.loc['Cholesterol, Total_1'], row.loc['HDL Cholesterol_1'], row.loc['SBP_1'], smoking, diabetes, bool(row.loc['Blood Pressure_1']), race, 10]
    score = frs(*X)

    risks.append(estimate_risk(score, mean, gender, race))

risks = np.array(risks)

'''

dataframe_risks = pd.DataFrame(risks)
dataframe_risks.columns = ["Risk"]

dataframe_risks = pd.concat([pt_ids.reset_index(drop = True), dataframe_risks, imaging_dates.reset_index(drop = True)], axis = 1)
dataframe_risks["Imaging_dt"] = pd.to_datetime(dataframe_risks["Imaging_dt"])
df_risks_first = dataframe_risks.groupby("Patient Id").apply(lambda x: x.sort_values("Imaging_dt").head(1)).reset_index(drop = True)
df_risks_last = dataframe_risks.groupby("Patient Id").apply(lambda x: x.sort_values("Imaging_dt", ascending = False).head(1)).reset_index(drop = True)
df_merged = df_risks_first.merge(df_risks_last, how = "inner", on = "Patient Id")
df_merged = df_merged.loc[df_merged['Imaging_dt_x'] != df_merged['Imaging_dt_y']]
df_merged['Time Delta'] = (df_merged['Imaging_dt_y'] - df_merged['Imaging_dt_x']).dt.days
df_merged = df_merged[['Time Delta', 'Risk_x', 'Risk_y']]
df_merged.columns = ['Time Delta (Days)', 'PCE_1', 'PCE_2']
df_merged.to_csv('./pce_risk_trajectories.csv', index = False)
'''

auroc = roc_auc_score(y[splits == 4], risks)
auprc = average_precision_score(y[splits == 4], risks)

data = (list(y[splits == 4]), list(risks))

rng = np.random.default_rng()
auroc_ci = bootstrap(data, roc_auc_score, vectorized = False, paired = True, random_state = rng)
auprc_ci = bootstrap(data, average_precision_score, vectorized = False, paired = True, random_state = rng)

print ("AUROC on the test set is {:.3f}".format(auroc))
print(auroc_ci.confidence_interval)
print ("AUPRC on the test set is {:.3f}".format(auprc))
print(auprc_ci.confidence_interval)
