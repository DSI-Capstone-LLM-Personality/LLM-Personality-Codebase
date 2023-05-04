import torch
import numpy as np
from functools import reduce
import random
from icecream import ic
from itertools import permutations
from collections import defaultdict
# External
from MPI.mpi import *


def entropy(arr): return -np.sum(arr * np.log(arr), axis=1)


def normalize(arr): return arr / np.sum(arr, axis=1, keepdims=True)


def mutual_information(x):
    x = torch.stack(x, dim=0).numpy()
    # ic(x.shape)
    # this is only for MPI 5 options templates
    assert len(x.shape) == 2 and x.shape[1] == 5
    x = normalize(x)
    h_y = entropy(np.mean(x, axis=0, keepdims=True)).item()
    h_y_given_x = np.mean(entropy(x))
    return h_y - h_y_given_x


class MIScorer():
    # This class is not necessary, remove later
    def __init__(self, ckpt_dir, ckpt_name):
        filename = ckpt_dir + ckpt_name
        self.ckpt = torch.load(filename)
        #

        # TODO: (Xiaoyang) this can be modified to be more generic...

    def __call__(self, x=None):
        if x is not None:
            print("Calculating Mutual Information of given probability vector...")
        else:
            x = torch.stack(self.ckpt.likelihood, dim=0).numpy()
            ic(x.shape)
            # this is only for MPI 5 options templates
            assert len(x.shape) == 2 and x.shape[1] == 5
            x = normalize(x)
        h_y = entropy(np.mean(x, axis=0, keepdims=True)).item()
        h_y_given_x = np.mean(entropy(x))
        return h_y - h_y_given_x


if __name__ == "__main__":
    # Simple test
    ckpt_dir = "checkpoint/mpis/Constraint/order-symmetry/bert-base-uncased/non-index/desc/"
    # ckpt_name = "[ocean_988]_[BERT|bert-base-uncased]_[non-index]_[order-III].pt"
    ckpt_name = "[ocean_988]_[BERT|bert-base-uncased]_[non-index]_[original].pt"

    mi_scorer = MIScorer(ckpt_dir, ckpt_name)
    mi = mi_scorer()
    print(mi)
