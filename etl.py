import os
import pickle
import pandas as pd
import torch
import numpy as np
import itertools
from torch.utils.data import TensorDataset, Dataset


def load_data(path, task, mode):
    task_dict = {1: "mortality/mortality.", 2: "readmission/readmission.", 3: "heart_failure/heart_failure.",
                 4: "sepsis/sepsis."}
    file_ids = open(path + task_dict[task] + 'ids.' + mode, 'rb')
    file_labels = open(path + task_dict[task] + 'labels.' + mode, 'rb')
    file_seqs = open(path + task_dict[task] + 'seqs.' + mode, 'rb')

    # dump information to that file
    data_ids = pickle.load(file_ids)
    data_labels = pickle.load(file_labels)
    data_seqs = pickle.load(file_seqs)
    return data_ids, data_labels, data_seqs


def calculate_num_features(seqs):
    nfeat = int(max(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(seqs))))) + 1)
    return nfeat


class VisitSeqLabelDataset(Dataset):
    def __init__(self, seqs, labels, num_features):
        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.labels = labels

        init_seq = []
        for s in seqs:
            temp_mat = np.zeros((len(s), num_features), dtype=int)
            for i, ix in enumerate(s):
                temp_mat[i][[int(j) for j in ix]] = 1
            init_seq.append(temp_mat)
        self.seqs = init_seq

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]


def dl_collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda tup: tup[0].shape[0], reverse=True)
    labels = []
    seqs = []
    lengths = []

    for seq, label in sorted_batch:
        seqs.append(seq)
        labels.append(label)
        lengths.append(seq.shape[0])

    max_length = np.max(lengths)

    new_seq = []
    for s in seqs:
        pad = max_length - len(s)
        temp_arr = np.array(s)
        pad_ar = np.pad(temp_arr, ((0, pad), (0, 0)))
        new_seq.append(pad_ar)

    seqs_tensor = torch.FloatTensor(np.array(new_seq))
    lengths_tensor = torch.LongTensor(np.array(lengths))
    labels_tensor = torch.LongTensor(np.array(labels))

    return (seqs_tensor, lengths_tensor), labels_tensor