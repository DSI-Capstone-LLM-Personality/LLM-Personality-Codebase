from email.policy import default
import torch
import numpy as np
from functools import reduce
import random
from icecream import ic
from itertools import permutations
from collections import defaultdict
# TODO: (Xiaoyang) Wrap all of these into a .yaml file later.
#####  OCEAN BASICS  #####
OCEAN = ['O', 'C', 'E', 'A', 'N']
# ------------------------------- #
# MPI IDX-ANSWER-SCORE CONVERSION #
# ------------------------------- #
#####  METADATA  #####
MPI_NUM_CHOICES = 5

#####  PROMPT TEMPLATE  #####
MPI_PROMPT = "Please choose from the following options to identify how accurately this statement describes you."
PROMPT_TEMPLATE = {'mpi-style': MPI_PROMPT}

######  LETTER ANSWER TEMPLATE  #####
LETTER_ONLY = np.array(['(A).', '(B).', '(C).', '(D).', '(E).'])
DESC_ONLY = np.array([
    "Very Accurate",
    "Moderately Accurate",
    "Neither Accurate Nor Inaccurate",
    "Moderately Inaccurate",
    "Very Inaccurate"])
LETTER_DESC = np.array(reduce(
    lambda lst, z: lst + [z[0] + " " + z[1]], zip(LETTER_ONLY, DESC_ONLY), []))

#####  WEIGHTED ANSWER TEMPLATE  #####
# SCORE_ONLY_POS = np.array([5, 4, 3, 2, 1], dtype=str)
# SCORE_ONLY_NEG = np.array([1, 2, 3, 4, 5], dtype=str)
SCORE_ONLY_POS = np.array(['(5).', '(4).', '(3).', '(2).', '(1).'], dtype=str)
SCORE_ONLY_NEG = np.array(['(1).', '(2).', '(3).', '(4).', '(5).'], dtype=str)
SCORE_DESC_POS = np.array(reduce(
    lambda lst, z: lst + [z[0] + " " + z[1]], zip(SCORE_ONLY_POS, DESC_ONLY), []))
SCORE_DESC_NEG = np.array(reduce(
    lambda lst, z: lst + [z[0] + " " + z[1]], zip(SCORE_ONLY_NEG, DESC_ONLY), []))

#####  ANSWER TEMPLATE #####
ANSWER_TEMPLATE = {
    'letter-only': LETTER_ONLY,
    'desc-only': DESC_ONLY,
    'letter-desc': LETTER_DESC,
    'score-only-pos': SCORE_ONLY_POS,
    'score-only-neg': SCORE_ONLY_NEG,
    'score-desc-pos': SCORE_DESC_POS,
    'score-desc-neg': SCORE_DESC_NEG
}
#####  SCORE DICTIONARY  #####
MPI_IDX_TO_SCORE_NEG = np.arange(1, 6, 1)
MPI_IDX_TO_SCORE_POS = np.arange(5, 0, -1)
MPI_SCORE = {
    "+": MPI_IDX_TO_SCORE_POS,
    "-": MPI_IDX_TO_SCORE_NEG
}


# TODO: MBTI? or other personality test
# -------------------------------- #
# MBTI IDX-ANSWER-SCORE CONVERSION #
# -------------------------------- #
# ......


# ----------------- #
# UTILITY FUNCTIONS #
# ----------------- #
# CHECKPOINT FILENAME FORMAT
# [dset]_[model | version]_[choice_type]_[ll_type]
def log_fname(dset, model_desc, answer_type, ll_type=None):
    family, version = model_desc['family'], model_desc['version']
    if ll_type is None:
        return f"[{dset}]_[{family}|{version}]_[{answer_type}]"
    return f"[{dset}]_[{family} | {version}]_[{answer_type}]_[{ll_type}]"


##### Deal with choice order selection #####
def shuffle_choice(choice_lst):
    assert type(
        choice_lst) == np.ndarray, 'Please format your choice list as numpy array.'
    assert len(np.shape(choice_lst)) == 1
    assert len(choice_lst) > 1
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


MPI_IDX_ORDERS = order_ranking_dict(permute_orders(MPI_NUM_CHOICES))


def ordered_lst_to_str(ordered_lst, style='mpi'):
    if style == 'mpi':
        option = "\nOptions: "
        for choice in ordered_lst:
            option += f"\n{choice} "  # Prepend \n as designed choice
        return option + "\nAnswers: "
    else:
        assert False, 'Unrecognized option style.'


class MPIQuestionFormatter():
    def __init__(self, prompt: str, options: dict):
        self.prompt = prompt
        self.option = options

    def __call__(self, statement, key):
        question = f"Given a statement of you: \"You {statement}.\" "
        return question + self.prompt + ordered_lst_to_str(self.option[key])


def line(n=40): print("-"*n)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Test case:
# ic(order_distance([5, 4, 3, 2, 1]))
# ic(order_distance([1, 2, 3, 4, 5]))
# ic(order_distance([1, 2, 5, 3, 4]))
# ic(permute_orders(5).shape)
orders = order_ranking_dict(permute_orders(5), True)
ic(np.array(orders[11])[np.random.choice(8, 1)])


# 4  | 2 (all)
# 5  | 4
# 6  | 14
# 7  | 32 # [1, 0, 4, 3, 2]
# 8  | 18
# 9  | 28 # [4, 0, 2, 3, 1]
# 10 | 14
# 11 | 8  # [3, 1, 4, 0, 2]
