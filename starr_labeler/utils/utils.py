import os
from itertools import chain

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupKFold


def frequency_by_num_patients(iterator, measure_type, plt_path=None, plt_num=30, fig_size=10):
    first = True
    for chunk in iterator:
        chunk_occ = pd.DataFrame(chunk.groupby(["Patient Id", measure_type]).head(1))
        chunk_occ = pd.DataFrame(chunk_occ[measure_type].value_counts())
        if first:
            freqs = chunk_occ
            first = False
        else:
            freqs = freqs.merge(chunk_occ, how="outer", left_index=True, right_index=True)
            freqs = freqs.fillna(0)
            freqs["NumPatients"] = freqs.sum(axis=1)
            freqs = pd.DataFrame(freqs.loc[freqs.NumPatients > 1, "NumPatients"])
            freqs = freqs.sort_values("NumPatients", ascending=False)
    if plt_path is not None:
        figsize = fig_size
        textsize = 1.5 * figsize
        plt.figure(figsize=(figsize, figsize))
        print(list(freqs[:plt_num].index))
        plt.bar(freqs[:plt_num].index, freqs[:plt_num].NumPatients)
        plt.xticks(rotation=90, size=textsize)
        plt.yticks(size=textsize)
        plt.ylabel("Number Of Patients", size=textsize)
        plt.tight_layout()
        plt.savefig(plt_path, bbox_inches="tight")
    return freqs


def visualize_progression(pat_data, lab_tests, pat_id, ct_dates=None):
    # print(pat_data.loc[pat_data.Result == lab_test])
    plt.figure(figsize=(7, 7))
    plt.bar(
        list(pat_data["Type"].value_counts().index[0:20]), pat_data["Type"].value_counts()[0:20]
    )
    plt.xticks(rotation=90)
    plt.ylabel("Number of Measurements")
    plt.savefig(f"./frequencies_pat_{pat_id}.png", bbox_inches="tight")
    num_lab_tests = len(lab_tests)

    fig, axs = plt.subplots(5, 4, figsize=(30, 30))
    fig_num = 0
    for i in range(5):
        for j in range(4):
            date_num = 0
            if fig_num == num_lab_tests:
                return
            lab_test = lab_tests[fig_num]
            fig_num += 1
            axs[i, j].plot_date(
                list(pat_data.loc[pat_data.Type == lab_test]["Event_dt"]),
                list(pat_data.loc[pat_data.Type == lab_test]["Value"]),
                color="b",
            )
            axs[i, j].set_title(lab_test)
            if ct_dates is not None:
                for date in ct_dates:
                    date_num += 1
                    if date_num == 1:
                        axs[i, j].axvline(date, color="r", label="CT Date")
                    else:
                        axs[i, j].axvline(date, color="r")
            axs[i, j].legend()
            axs[i, j].set_ylabel("Value")
    plt.savefig(f"./trajectories_pat_{pat_id}.png", bbox_inches="tight")


def merge_dfms(dfs):
    if len(dfs) == 1:
        return dfs[0]
    else:
        for i in range(len(dfs) - 1):
            if i == 0:
                input_features = dfs[i].merge(
                    dfs[i + 1], how="outer", on=["Patient Id", "Accession Number"]
                )
            else:
                input_features = input_features.merge(
                    dfs[i + 1], how="outer", on=["Patient Id", "Accession Number"]
                )
        return input_features


def data_iterator(base_path, file_name, use_cols, chunksize=1000000):
    iterators = []
    directories = os.listdir(base_path)
    if file_name in directories:
        iterators.append(
            pd.read_csv(
                os.path.join(base_path, file_name),
                chunksize=chunksize,
                usecols=use_cols,
                low_memory=False,
            )
        )
    else:
        for dir_name in [str(1), str(2), str(3)]:
            iterators.append(
                pd.read_csv(
                    os.path.join(base_path, dir_name, file_name),
                    chunksize=chunksize,
                    usecols=use_cols,
                    low_memory=False,
                )
            )
    return chain(*iterators)


class patient_iterator:
    def __init__(self, base_path, file_name, use_cols, chunksize=1000000):
        self.base_path = base_path
        self.file_name = file_name
        self.chunksize = chunksize
        self.stop = False
        self.use_cols = use_cols

    def __iter__(self):
        self.patient_idx = 0
        self.my_data_iterator = iter(
            data_iterator(self.base_path, self.file_name, self.use_cols, self.chunksize)
        )
        self.curr_data_chunk = next(self.my_data_iterator)
        self.unique_ids = self.curr_data_chunk.loc[:, "Patient Id"].unique()
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration
        reserve_chunk = self.curr_data_chunk.loc[
            self.curr_data_chunk["Patient Id"] == self.unique_ids[-1]
        ]
        to_return = self.curr_data_chunk[self.curr_data_chunk["Patient Id"] != self.unique_ids[-1]]
        try:
            self.curr_data_chunk = pd.concat([reserve_chunk, next(self.my_data_iterator)])
        except BaseException:
            self.stop = True
            to_return = pd.concat([to_return, reserve_chunk])
            if to_return.empty:
                raise StopIteration from None
        self.unique_ids = self.curr_data_chunk.loc[:, "Patient Id"].unique()
        return to_return


class one_patient_iterator:
    def __init__(self, base_path, file_name, use_cols, chunksize=1000000):
        self.base_path = base_path
        self.file_name = file_name
        self.chunksize = chunksize
        self.use_cols = use_cols

    def __iter__(self):
        self.patient_idx = 0
        self.my_data_iterator = iter(
            data_iterator(self.base_path, self.file_name, self.use_cols, self.chunksize)
        )
        self.curr_data_chunk = next(self.my_data_iterator)
        self.unique_ids = self.curr_data_chunk.loc[:, "Patient Id"].unique()
        return self

    def __next__(self):
        pt_id = self.unique_ids[self.patient_idx]
        if self.patient_idx == self.unique_ids.shape[0] - 1:
            reserve_chunk = self.curr_data_chunk.loc[self.curr_data_chunk["Patient Id"] == pt_id]
            self.curr_data_chunk = next(self.my_data_iterator)
            self.curr_data_chunk = pd.concat([reserve_chunk, self.curr_data_chunk])
            self.unique_ids = self.curr_data_chunk.loc[:, "Patient Id"].unique()
            self.patient_idx = 0
            pt_id = self.unique_ids[self.patient_idx]
        to_return = self.curr_data_chunk.loc[self.curr_data_chunk["Patient Id"] == pt_id]
        self.patient_idx += 1
        return to_return


def get_splits(splits, labels):
    labels = labels.reset_index(drop=True)
    X = labels["Label"]
    y = labels["Label"]
    groups = labels["Patient Id"]
    group_kfold = GroupKFold(n_splits=splits)
    group_kfold.get_n_splits(X, y, groups)
    split_idx = 0
    for _, test_index in group_kfold.split(X, y, groups):
        labels.loc[test_index, "Split"] = split_idx
        split_idx += 1
    return labels
