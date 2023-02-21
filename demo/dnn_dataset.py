import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.preprocessing import normalize

class dnn_dataset(Dataset):
    def __init__(self, mode, inputs, labels):
        features = pd.read_csv(inputs)
        y = pd.read_csv(labels)
        data = features.merge(y, how = 'inner', on = ['Patient Id', 'Accession Number'])
        zero_or_one = np.logical_or(data.iloc[:, -2] == 0, data.iloc[:, -2] == 1)
        data = data.loc[zero_or_one, :]

        if mode == 'train':
            data = data.loc[data['Split'] < 3]

        if mode == 'eval':
            data = data.loc[data['Split'] == 3]

        if mode == 'test':
            data = data.loc[data['Split'] == 4]

        unnormed = np.array(data.iloc[:, 2:-2])
        self.features = (unnormed - unnormed.mean(axis = 0)) / (unnormed.std(axis = 0) + 1)
        y = data.iloc[:, -2]
        self.y = np.array(y)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx, :], self.y[idx]