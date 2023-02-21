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

from Stanford_Extract import generate_inputs, get_parser, setup_cfg

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
X = pd.read_csv('../Stanford_Extract/checkpoints/inputs.csv')
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
        #'gamma': [0.5, 1, 1.5, 2, 5],
        #'subsample': [0.6, 0.8, 1.0],
        #'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [5, 6, 7],
        'n_estimators': [200, 250, 300]
        }

fit_params = fit_params ={
        "early_stopping_rounds":42, 
        "eval_metric" : "mae", 
        "eval_set" : [[testX, testY]]
        }

xgb = xgb.XGBClassifier(learning_rate = 0.05, objective = 'binary:logistic', nthread = 1, eval_metric = 'logloss', use_label_encoder = False)

grid_search = GridSearchCV(xgb, param_grid = params, refit = 'roc_auc', scoring = ['roc_auc', 'average_precision'], n_jobs = 4, cv = cvsplit_iter, verbose = 3)

grid_search.fit(x_vals, y_vals)