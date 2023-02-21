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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import bootstrap

from grid_search_early_stopping import *

class CVSplits:
    def __init__(self, num_splits, splits):
        self.max = num_splits
        self.splits = splits

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            indices = (list(np.argwhere(np.logical_and(self.splits != self.n, self.splits < self.max))[:, 0]), list(np.argwhere(self.splits == self.n)[:, 0]))
            self.n += 1
            return indices
        else:
            raise StopIteration

path = Path(sys.path[0])
sys.path.insert(0, os.path.join(str(path.parent.absolute()), "Stanford_Extract"))
sys.path.insert(0, os.path.join(str(path.parent.absolute()), "Stanford_Extract/features"))
sys.path.insert(0, str(path.parent.absolute()))

from Stanford_Extract import compute_features, get_parser, setup_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--eta', type = float, default = 0.05)
parser.add_argument('--max_depth', type = int, default = 6)
parser.add_argument('--num_parallel_tree', type = int, default = 1)
parser.add_argument('--num_iterations', type = int, default = 300)

args_xgb = parser.parse_args()

config_file_path = '../configs/xgboost.yaml'

args = get_parser().parse_args()
args.config_file = config_file_path
cfg = setup_cfg(args)

wandb.login(key = 'b99f216f5cf64b99d55527e8c307c9858063cbbe')
wandb.init(project = "feb_22_cohort_ehr_processing", group = "xgboost", config = cfg)
wandb.config.update(args_xgb)

num_splits = 5

#X = generate_inputs(config_file_path, save_path = '../Stanford_Extract/checkpoints/inputs.csv')
#X = pd.read_csv('../Stanford_Extract/checkpoints_v1/inputs.csv')
X = pd.read_csv('../Stanford_Extract/checkpoints/inputs_pce_v1.csv')
print(X.columns)
X = X.drop(labels = 'Imaging_dt', axis = 1)
X['Patient Id'] = X['Patient Id'].astype("string").str.replace(r'^(0+)', '', regex=True).fillna('0')
X['Accession Number'] = X['Accession Number'].astype("string").str.replace(r'^(0+)', '', regex=True).fillna('0')

y = pd.read_csv('../Stanford_Extract/checkpoints/labels.csv')
y['Patient Id'] = y['Patient Id'].astype("string").str.replace(r'^(0+)', '', regex=True).fillna('0')
y['Accession Number'] = y['Accession Number'].astype("string").str.replace(r'^(0+)', '', regex=True).fillna('0')

data = X.merge(y, how = 'inner', on = ['Patient Id', 'Accession Number'])

print(data)

wandb.config.update({"Input Shape": data.shape})

x_vals = data.iloc[:, 2:-2]
y_vals = data.iloc[:, -2]
splits = data.iloc[:, -1]

zero_or_one = (y_vals == 0) | (y_vals == 1)
x_vals = np.array(x_vals.loc[zero_or_one])
y_vals = np.array(y_vals.loc[zero_or_one]).astype(int)
splits = np.array(splits.loc[zero_or_one])

cvsplit_class = CVSplits(4, splits)
cvsplit_iter = iter(cvsplit_class)

grid_params = {
        #'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.5, 1],
        #'subsample': [0.6, 0.8, 1.0],
        #'colsample_bytree': [0.6, 0.8, 1.0],
        'objective': ['binary:logistic'],
        'eval_metric': ['logloss'],
        'use_label_encoder': [False],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.03, 0.05, 0.07],
        'n_estimators': [300]
        }

fit_params = {
        "early_stopping_rounds": 100, 
        "eval_metric" : "auc", 
        "verbose": 0
        }

if True:
    scorer = make_scorer(roc_auc_score, greater_is_better = True, needs_threshold = True)
    best_params_xgb = GridSearchCV_XGB_early_stoppping(grid_params, fit_params, scorer, cvsplit_iter, x_vals, y_vals)

else:
    best_params_xgb = np.load('best_params_xgb.npy', allow_pickle=True).item()

best_xgb = XGBClassifier(**best_params_xgb)
best_xgb.fit(x_vals[splits < 3], y_vals[splits < 3], verbose = 0)
y_pred = best_xgb.predict_proba(x_vals[splits == 4])[:, 1]

auroc = roc_auc_score(y_vals[splits == 4], y_pred)
auprc = average_precision_score(y_vals[splits == 4], y_pred)

data = (list(y_vals[splits == 4]), list(y_pred))

rng = np.random.default_rng()
auroc_ci = bootstrap(data, roc_auc_score, vectorized = False, paired = True, random_state = rng)
auprc_ci = bootstrap(data, average_precision_score, vectorized = False, paired = True, random_state = rng)

print ("AUROC on the test set is {:.3f}".format(auroc))
print(auroc_ci.confidence_interval)
print ("AUPRC on the test set is {:.3f}".format(auprc))
print(auprc_ci.confidence_interval)

np.save('best_params_xgb.npy', best_params_xgb)