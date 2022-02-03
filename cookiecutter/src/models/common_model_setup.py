#!/usr/bin/env python3

print("Importing packages")

import sys
import glob
import os
import os.path

import math
import torch
from torch import nn, Tensor, sigmoid, tanh, relu, softmax
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import matthews_corrcoef as mcc
import copy
import numpy as np

import kmbio
from kmbio import PDB
from Bio import SeqIO
import torch_geometric

sys.path.append("../PS")
import ProteinSolver as proteinsolver

print("Functions..")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True

data_folder = "../../../data/"

class EpitopeDataset(data.Dataset):
    def __init__(self,embeddings,labels,mask,batch=4):
        self.embeddings = embeddings
        self.labels = labels
        self.mask = mask

    def __len__(self):
        return (self.embeddings).shape[0]

    def __getitem__(self,idx):
        return self.embeddings[idx] , self.labels[idx], self.mask[idx]

def create_embedding_dicts(solved):
    embedding_dicts = list()

    # This is due to the way we created the embeddings originally
    # since they were created with different versions of the same scripts.
    # This hotfix should be eaasily fixed by rerunning. Let's see if it happens!
    if solved:
        SEQ_KEY_NAME = "seq_pdb"
        embedding_type = "../../../data/processed/cleaned_solved_ps_embeddings/"
    else:
        SEQ_KEY_NAME = "seq"
        embedding_type = "../../../data/processed/af2_ps_embeddings/"

    for pt in glob.glob(embedding_type+'*.pt'):
        pt_obj = torch.load(pt)
        embedding_dicts.append(pt_obj)

    fasta = {}
    with open("../../../data/raw/antigens_before_clustering.fasta", "r") as file_one:
        for line in file_one:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                active_sequence_name = line[1:]
                if active_sequence_name not in fasta:
                    fasta[active_sequence_name] = []
                continue
            sequence = line
            fasta[active_sequence_name].append(sequence)

    for embed_dict in embedding_dicts:
        assert len(embed_dict[SEQ_KEY_NAME]) == len(fasta[embed_dict['pdb_id']][0])
        embed_dict[SEQ_KEY_NAME] = fasta[embed_dict['pdb_id']][0]

    return embedding_dicts, SEQ_KEY_NAME

def create_sequential_datasets(list_of_dicts, SEQ_KEY_NAME, device):
    n_seqs = len(list_of_dicts)
    seq_max_length = max([len(embed_dict[SEQ_KEY_NAME]) for embed_dict in list_of_dicts])
    embedding_dim = 128

    with open("../../../data/interim/postapr2018_antigens.test.txt", "r") as infile:
        test_names = set([l.strip() for l in infile.readlines()])
    with open("../../../data/interim/preapr2018_antigens.train.txt", "r") as infile:
        train_names = set([l.strip() for l in infile.readlines()])
    with open("../../../data/interim/preapr2018_antigens.validation.txt", "r") as infile:
        val_names = set([l.strip() for l in infile.readlines()])

    X = torch.zeros(size=(n_seqs, seq_max_length, embedding_dim))
    y = torch.zeros(size=(n_seqs, seq_max_length))
    mask = torch.zeros(size=(n_seqs, seq_max_length))
    test_idx = list()
    train_idx = list()
    val_idx = list()
    for i, embed_dict in enumerate(list_of_dicts):
        seq = embed_dict[SEQ_KEY_NAME]
        pdb_id = embed_dict['pdb_id']
        embedding = embed_dict['per_tok'].detach()

        if pdb_id in test_names:
            test_idx.append(i)
        elif pdb_id in train_names:
            train_idx.append(i)
        elif pdb_id in val_names:
            val_idx.append(i)
        y_ = torch.Tensor([1 if letter.isupper() else 0 for letter in seq])

        X[i, 0:embedding.shape[0]] = embedding
        y[i, 0:len(y_)] = y_
        mask[i, 0:len(y_)] = torch.ones((1,len(y_)))

    test_idx = np.asarray(test_idx)
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)


    random_indices = np.random.choice(len(train_idx), len(train_idx), replace=False)

    inputs_train = X[train_idx, :, :]
    inputs_test = X[test_idx, :, :]
    inputs_val = X[val_idx, :, :]
    targets_train = y[train_idx]
    targets_test = y[test_idx]
    targets_val = y[val_idx]
    masks_train = mask[train_idx]
    masks_test = mask[test_idx]
    masks_val = mask[val_idx]

    print("Training dims:", inputs_train.shape, targets_train.shape)
    print("Test dims:", inputs_test.shape, targets_test.shape)
    print("Validation dims:", inputs_val.shape, targets_val.shape)

    training_set = EpitopeDataset(inputs_train.to(device), targets_train.to(device), masks_train.to(device))
    validation_set = EpitopeDataset(inputs_val.to(device), targets_val.to(device), masks_val.to(device))
    test_set = EpitopeDataset(inputs_test.to(device), targets_test.to(device), masks_test.to(device))

    return training_set, validation_set, test_set

def create_positional_datasets(list_of_dicts, SEQ_KEY_NAME, device):
    with open("../../../data/interim/postapr2018_antigens.test.txt", "r") as infile:
        test_names = set([l.strip() for l in infile.readlines()])
    with open("../../../data/interim/preapr2018_antigens.train.txt", "r") as infile:
        train_names = set([l.strip() for l in infile.readlines()])
    with open("../../../data/interim/preapr2018_antigens.validation.txt", "r") as infile:
        val_names = set([l.strip() for l in infile.readlines()])

    test_idx = []
    train_idx = []
    val_idx = []

    # Define partition sizes
    y = torch.Tensor()
    X = torch.Tensor()
    for embed_dict in list_of_dicts:
        seq = embed_dict[SEQ_KEY_NAME]
        y_ = torch.Tensor([1 if letter.isupper() else 0 for letter in seq])
        #print(seq, y)

        pdb_id = embed_dict['pdb_id']

        if pdb_id in test_names:
            test_idx.extend(list(range(len(y), len(y)+len(y_))))
        elif pdb_id in train_names:
            train_idx.extend(list(range(len(y), len(y)+len(y_))))
        elif pdb_id in val_names:
            val_idx.extend(list(range(len(y), len(y)+len(y_))))


        embedding = embed_dict['per_tok'].detach()
        X = torch.cat((X, embedding), 0)
        y = torch.cat((y, y_), 0)

    inputs_train = X[train_idx, :]
    inputs_val = X[val_idx, :]
    inputs_test = X[test_idx, :]
    targets_train = y[train_idx]
    targets_val = y[val_idx]
    targets_test = y[test_idx]

    # Get inputs and targets for each partition

    print("Training dims:", inputs_train.shape, targets_train.shape)
    print("Test dims:", inputs_test.shape, targets_test.shape)
    print("Validation dims:", inputs_val.shape, targets_val.shape)

    training_set = data.TensorDataset(inputs_train.to(device), targets_train.to(device))
    validation_set = data.TensorDataset(inputs_val.to(device), targets_val.to(device))
    test_set = data.TensorDataset(inputs_test.to(device), targets_test.to(device))

    return training_set, validation_set, test_set


