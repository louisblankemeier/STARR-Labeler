import os
# import pytorch dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import sys

# write a dataloader that takes in a csv with columns Patient Id, Date, Input for inference
class InferenceDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_id = self.data.iloc[idx, 0]
        date = self.data.iloc[idx, 1]
        input = self.data.iloc[idx, 2]
        sample = {'patient_id': patient_id, 'date': date, 'input': input}

        return sample

class pmb_classifier(nn.Module):
    def __init__(self):
        super(pmb_classifier, self).__init__()
        self.pmb_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, x):
        hiddens = self.pmb_model(x).last_hidden_state
        x = self.classifier(hiddens[:, 0, :])
        return x

# now write a class to perform inference
class Inference:
    def __init__(self, model_path, dataset_path, output_path, batch_size=128):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        self.model = pmb_classifier()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.dataset = InferenceDataset(dataset_path)
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.output_path = output_path

    def __call__(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        patient_ids = []
        dates = []
        inputs = []
        outputs = []
        probs = []
        with torch.no_grad():
            for i_batch, sample_batched in tqdm(enumerate(dataloader)):
                tokens = self.tokenizer(list(sample_batched['input']), padding="max_length", max_length=128, truncation=True)
                input_ids = tokens['input_ids']
                input_ids = torch.Tensor(input_ids).type(torch.int)
                input_ids = input_ids.to(self.device)
                output = torch.sigmoid(self.model(input_ids))[:, 0]
                probs.extend(list(1 - output.cpu().numpy()))
                output = (output < 0.1).int()
                outputs.extend(list(output.cpu().numpy()))
                patient_ids.extend(list(sample_batched['patient_id'].numpy()))
                dates.extend(sample_batched['date'])
                inputs.extend(sample_batched['input'])
        df = pd.DataFrame({'Patient Id': patient_ids, 'Date': dates, 'Input': inputs, 'Output': outputs, 'Probability': probs})
        # date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        # sort by patient id and date
        df = df.sort_values(by=['Patient Id', 'Date'])
        df.to_csv(self.output_path, index=False)





    
    