import re
import os
import sys
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import re
from negspacy import termsets
import hydra
from omegaconf import DictConfig
import logging

from hyde_inference import Inference

class mcm_generator():
    def __init__(self, config):
        self.cfg = config
        self.hyde_cfg = config['HyDE']
        self.filter_regex = self.hyde_cfg['FILTER_REGEX']
        self.mask_replacement_regex = self.hyde_cfg['MASK_REPLACEMENT_REGEX']
        self.disease_name = config['disease_name']
        # self.cross_walk_path = "/bmrNAS/scandata/abct_ehr/priority_crosswalk_all.csv"
        
    def _load_notes(self):
        """Load clinical notes from csv file
        """
        if not os.path.exists(self.hyde_cfg.OUTPUT_DIR):
            os.makedirs(self.hyde_cfg.OUTPUT_DIR)

        # crosswalk = pd.read_csv(self.cross_walk_path)
        # crosswalk['accession'] = crosswalk['accession'].astype(str)
        # crosswalk = crosswalk.rename(columns={"accession": "Accession Number"})

        clinical_notes0 = pd.DataFrame()
        logging.info("Ingesting clinical notes")

        for idx in range(1,4):
            idx_path = f"{idx}/clinical_note.csv"
            #idx_path = f"{idx}/radiology_report.csv"

            clinical_notes0 = pd.concat([clinical_notes0, pd.read_csv(os.path.join(self.hyde_cfg.CLINICAL_NOTES_PATH, idx_path))])
            logging.info(f"Clinical_notes_shape: {clinical_notes0.shape}")

        # clinical_notes0['Accession Number'] = clinical_notes0['Accession Number'].astype(str)
        # clinical_notes0 = clinical_notes0.merge(crosswalk, on = "Accession Number", how = "inner")

        clinical_notes0 = clinical_notes0[["Patient Id", "Date", "Type", "Title", "Text"]]

        logging.info(f"Number of unique patients before note type filtering: {len(clinical_notes0['Patient Id'].unique())}")
        logging.info(f"Number of clinical notes before note type filtering: {len(clinical_notes0)}")
        logging.info(f"Note types: {self.hyde_cfg.NOTE_TYPES}")
        logging.info(f"Note type value counts: {clinical_notes0['Type'].value_counts()}")

        correct_type = np.array(clinical_notes0['Type'].str.contains("|".join(self.hyde_cfg.NOTE_TYPES), case = False, na = False))
        clinical_notes0 = clinical_notes0[correct_type]

        logging.info(f"Number of unique patients after note type filtering: {len(clinical_notes0['Patient Id'].unique())}")
        logging.info(f"Number of clinical notes after note type filtering: {len(clinical_notes0)}")

        return clinical_notes0


    def generate_mcm_json(self):
        """Process clinical notes to generate masked contextual mentions
        """ 
        clinical_notes0 = self._load_notes()

        num_processes = 32
        processes = []
        results = []
        num_per_process = int(clinical_notes0.shape[0] / num_processes)
        before_terms = "|".join(self.hyde_cfg.BEFORE_TERMS)
        after_terms = "|".join(self.hyde_cfg.AFTER_TERMS)

        i = 0
        inputs_to_map = []
        while (num_per_process * i) < clinical_notes0.shape[0]:
            start_idx = num_per_process * i
            if (num_per_process * (i + 1)) > clinical_notes0.shape[0]: 
                end_idx = clinical_notes0.shape[0]
            else:
                end_idx = num_per_process * (i + 1)
            input_notes = clinical_notes0.iloc[start_idx:end_idx, :]
            inputs_to_map.append((input_notes, before_terms, after_terms, self.hyde_cfg.MASK_REPLACEMENT_REGEX, self.hyde_cfg.FILTER_REGEX))
            i += 1

        number_of_cores = cpu_count()

        logging.info("Starting to generate masked contextual mentions")
        logging.info(f"Number of cores per task {number_of_cores}")

        # num_mcms, num_notes_with_mcms, size_of_notes_with_mcms
        pool = Pool(num_processes)
        results = pool.map(self._process_clinical_notes, inputs_to_map)
        full_contains = []
        num_notes = []
        num_mcms = []
        size_of_notes_with_mcms = []
        for result in results:
            full_contains.extend(result[0])
            num_mcms.append(result[1])
            num_notes.append(result[2])
            size_of_notes_with_mcms.extend(result[3])

        # log the length of size of notes with mcms
        logging.info(f"Length of size of notes with mcms: {len(size_of_notes_with_mcms)}")
        logging.info(f"Sum of mentions per note: {np.sum(num_mcms)}")
        logging.info(f"Sum of number of notes with mentions: {np.sum(num_notes)}")
        logging.info(f"Average size of notes with mentions: {np.mean(size_of_notes_with_mcms)}")

        final_dict = {}
        final_dict['mentions'] = full_contains

        with open(os.path.join(self.hyde_cfg.OUTPUT_DIR, self.hyde_cfg.OUTPUT_FILE_NAME + ".json"), "w") as outfile:
            json.dump(final_dict, outfile, indent=4)


    def _process_clinical_notes(self, inputs):
        """Process clinical notes to generate masked contextual mentions
        """
        clinical_notes = inputs[0]
        before_terms = inputs[1]
        after_terms = inputs[2]
        mask_replacement_regex = inputs[3]
        sort_string = inputs[4]
        num_clinical_notes = clinical_notes.shape[0]
        note_list = []
        num_mcms = 0
        num_notes_with_mcms = 0
        size_of_notes_with_mcms = []

        for i in range(num_clinical_notes): 
            if (i % 1000) == 0:
                # log iteration for process
                logging.info(f"Iteration {i} for process {os.getpid()}")
                sys.stdout.flush()
            note = str(clinical_notes.iloc[i, 4])
            note = re.sub(sort_string, lambda x: x.group().replace(" ", "_"), note)
            split_notes = np.array(note.split()) 
            positive_idx = [i for i, item in enumerate(split_notes) if re.search(mask_replacement_regex, item, re.IGNORECASE)]
            num_mcms += len(positive_idx)
            num_notes_with_mcms += 1 if len(positive_idx) > 0 else 0

            if len(positive_idx) > 0:
                size_of_notes_with_mcms.append(len(split_notes))
            for j in positive_idx:
                start_idx = j - 32
                if start_idx < 0:
                    start_idx = 0
                end_idx = j - 1
                if end_idx < 0:
                    left_context = ""
                else:
                    if ((j - 2) >= 0) and re.search(before_terms, split_notes[j - 1], re.IGNORECASE):
                        left_context = split_notes[start_idx:j - 1]
                    elif ((j - 3) >= 0) and re.search(before_terms, split_notes[j - 2] + " " + split_notes[j - 1], re.IGNORECASE):
                        left_context = split_notes[start_idx:j - 2]
                    else:
                        left_context = split_notes[start_idx:j]
                mention = split_notes[j]
                end_idx = j + 32 + 1
                if end_idx >= len(split_notes):
                    end_idx = len(split_notes) - 1
                start_idx = j + 1
                if start_idx >= len(split_notes):
                    right_context = ""
                else:
                    if ((start_idx + 1) < len(split_notes)) and re.search(after_terms, split_notes[start_idx], re.IGNORECASE):
                        right_context = split_notes[start_idx + 1: end_idx]
                    elif ((start_idx + 2) < len(split_notes)) and re.search(after_terms, split_notes[start_idx] + " " + split_notes[start_idx + 1], re.IGNORECASE):
                        right_context = split_notes[start_idx + 2: end_idx]
                    else:
                        right_context = split_notes[start_idx: end_idx]

                note_dict = {}
                note_dict['Patient Id'] = str(clinical_notes.iloc[i, 0])
                note_dict['Date'] = clinical_notes.iloc[i, 1]
                note_dict['Type'] = clinical_notes.iloc[i, 2]
                note_dict['Title'] = clinical_notes.iloc[i, 3]
                left_context = " ".join(left_context)
                note_dict['Left Context'] = left_context
                note_dict['Mention'] = mention
                right_context = " ".join(right_context)
                note_dict['Right Context'] = right_context
                note_dict['Label'] = str(-1)
                note_list.append(note_dict)

        return note_list, num_mcms, num_notes_with_mcms, size_of_notes_with_mcms


    def generate_mcm_dataframe(self):
        """Generate dataframe of masked contextual mentions for use as input to model
        """
        f = open(os.path.join(self.hyde_cfg.OUTPUT_DIR, self.hyde_cfg.OUTPUT_FILE_NAME + ".json"), "r")
        data = json.load(f)

        inputs = []
        labels = []
        patient_ids = []
        dates = []
        local = []
        for mention in data['mentions']:
            inputs.append(mention['Left Context'] + " [MASK] " + mention['Right Context'])
            local.append(mention['Left Context'][-1 * int(self.hyde_cfg.CONTEXT_SIMILARITY_CHARS):-1] + " " + mention['Right Context'][0:int(self.hyde_cfg.CONTEXT_SIMILARITY_CHARS)])
            labels.append(mention['Label'])
            patient_ids.append(mention['Patient Id'])
            dates.append(mention['Date'])

        dataframe = pd.DataFrame(list(zip(patient_ids, dates, inputs, local)), columns = ['Patient Id', 'Date', 'Input', 'Local'])
        dataframe['Patient Id'] = pd.to_numeric(dataframe['Patient Id'])
        dataframe['Date'] = pd.to_datetime(dataframe['Date'], utc = True)
        logging.info(f"Number of mentions in dataframe before dropping local_duplicates: {len(dataframe)}")

        dataframe = dataframe.drop_duplicates(subset = ['Local'])
        dataframe = dataframe[["Patient Id", "Date", "Input"]]
        logging.info(f"Number of mentions in dataframe after dropping local duplicates: {len(dataframe)}")

        dataframe.to_csv(os.path.join(self.hyde_cfg.OUTPUT_DIR, self.hyde_cfg.OUTPUT_FILE_NAME + ".csv"), index = False)

    def run_hyde(self):
        """Run hyde
        """
        if not os.path.exists(os.path.join(self.hyde_cfg.OUTPUT_DIR, self.hyde_cfg.OUTPUT_FILE_NAME + ".json")):
            self.generate_mcm_json()

        if not os.path.exists(os.path.join(self.hyde_cfg.OUTPUT_DIR, self.hyde_cfg.OUTPUT_FILE_NAME + ".csv")):
            self.generate_mcm_dataframe()

        model_path = '/dataNAS/people/lblankem/starr_labeler/starr_labeler/labels/models/rs_c2_c3_a1_a2_a3_a4_bst.pt'
        dataset_path = os.path.join(self.hyde_cfg.OUTPUT_DIR, self.hyde_cfg.OUTPUT_FILE_NAME + ".csv")
        output_path = os.path.join(self.hyde_cfg.OUTPUT_DIR, self.hyde_cfg.OUTPUT_FILE_NAME + "_predictions.csv")
        inference = Inference(model_path, dataset_path, output_path)
        inference()

        