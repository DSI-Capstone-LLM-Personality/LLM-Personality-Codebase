import torch
import numpy as np
from functools import reduce
import random
from icecream import ic
from itertools import permutations
from collections import defaultdict
from Model.template import *

######  OCEAN BASICS  ######
OCEAN = ['O', 'C', 'E', 'A', 'N']
# TODO: MBTI? or other personality test

# ----------------------------------------------------- #
#                  UTILITY FUNCTIONS                    #
# ----------------------------------------------------- #
######  CHECKPOINT FILENAME FORMAT  ######
# [dset]_[model | version]_[choice_type]_[ll_type]


def log_fname(dset, model_desc, answer_type, ll_type=None):
    family, version = model_desc['family'], model_desc['version']
    if ll_type is None:
        return f"[{dset}]_[{family}|{version}]_[{answer_type}]"
    return f"[{dset}]_[{family} | {version}]_[{answer_type}]_[{ll_type}]"


######  RANDOM ORDER SELECTION -- SYMMETRY EXPERIMENT  ######
def shuffle_choice(choice_lst, given_idx=None):
    assert type(
        choice_lst) == np.ndarray, 'Please format your choice list as numpy array.'
    assert len(np.shape(choice_lst)) == 1
    assert len(choice_lst) > 1
    if given_idx is not None:
        return choice_lst[given_idx], given_idx
    n = len(choice_lst)
    rand_idx = np.random.choice(n, n, replace=False)
    return choice_lst[rand_idx], rand_idx


def order_distance(idx):
    lst1 = np.array([idx[0]] + list(idx))
    lst2 = np.array(list(idx) + [idx[-1]])
    return np.sum(np.abs(lst1 - lst2))


def permute_orders(n):
    return np.array(list(permutations(np.arange(n))))


def order_ranking_dict(permutation, verbose=False):
    orders = defaultdict(list)
    for order in permutation:
        orders[order_distance(order)].append(order)
    if verbose:
        for key in orders:
            print(f"{key:<2} | {len(orders[key])}")
    return orders


# MPI_IDX_ORDERS = order_ranking_dict(permute_orders(MPI_NUM_CHOICES))


def ordered_lst_to_str(ordered_lst, style='mpi'):
    if style == 'mpi':
        option = "\nOptions: "
        for choice in ordered_lst:
            option += f"\n{choice} "  # Prepend \n as designed choice
        return option + "\nAnswers: "
    else:
        assert False, 'Unrecognized option style.'

# Simple test cases:
# ic(order_distance([5, 4, 3, 2, 1]))
# ic(order_distance([1, 2, 3, 4, 5]))
# ic(order_distance([1, 2, 5, 3, 4]))
# ic(permute_orders(5).shape)
# orders = order_ranking_dict(permute_orders(5), True)
# ic(np.array(orders[11])[np.random.choice(8, 1)])

# order selection result
# 4  | 2 (all)
# 5  | 4
# 6  | 14
# 7  | 32 # [1, 0, 4, 3, 2]
# 8  | 18
# 9  | 28 # [4, 0, 2, 3, 1]
# 10 | 14
# 11 | 8  # [3, 1, 4, 0, 2]

###### FORMATTERS  ######


class MPIOptionFormatter():
    def __init__(self, index, desc):
        self.index = index
        self.desc = desc

    def __call__(self, order, shuffle_both=None):
        if order is not None:
            assert shuffle_both is not None
            if shuffle_both and self.index is not None:
                self.index = {k: v[order] for k, v in self.index.items()}
            # ic(self.desc)
            self.desc = {k: v[order] for k, v in self.desc.items()}
        if self.index is not None:
            return {k: concat(self.index[k], self.desc[k]) for k in ['+', '-']}
        else:
            return self.desc


def MPI_options_to_answers(index, desc, option, ans_type, order=None):
    if ans_type == 'desc':
        return desc
    elif ans_type == 'index':
        return index
    elif ans_type == 'index-desc':
        assert order is not None
        return {k: option[k][order] for k in ['+', '-']}
    else:
        assert False, 'Unrecognized answer template type.'


class MPIQuestionFormatter():
    def __init__(self, prompt: str, options: dict):
        self.prompt = prompt
        self.option = options

    def __call__(self, statement, key):
        question = f"Given a statement of you: \"You {statement}.\" "
        return question + self.prompt + ordered_lst_to_str(self.option[key])


######  UTILITY FUNCTIONS  ######
def line(n=40): print("-"*n)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
