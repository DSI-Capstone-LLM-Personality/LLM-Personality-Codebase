from aiohttp import AsyncIterablePayload
from sympy import Order
import torch
import numpy as np
from functools import reduce
import random
from icecream import ic
from itertools import permutations
from collections import defaultdict


def concat(index: np.ndarray, desc: np.ndarray):
    return np.array(reduce(
        lambda lst, z: lst + [z[0] + " " + z[1]], zip(index, desc), []))


##### MPI constants #####
MPI_NUM_CHOICES = 5
#####  PROMPT TEMPLATE  #####
MPI_PROMPT = "Please choose from the following options to identify how accurately this statement describes you."
PROMPT_TEMPLATE = {'mpi-style': MPI_PROMPT}

#####  LETTER ANSWER TEMPLATE  #####
LETTER_ONLY = np.array(['(A).', '(B).', '(C).', '(D).', '(E).'])
DESC_ONLY = np.array([
    "Very Accurate",
    "Moderately Accurate",
    "Neither Accurate Nor Inaccurate",
    "Moderately Inaccurate",
    "Very Inaccurate"])
LETTER_DESC = concat(LETTER_ONLY, DESC_ONLY)

#####  WEIGHTED ANSWER TEMPLATE  #####
# SCORE_ONLY_POS = np.array([5, 4, 3, 2, 1], dtype=str)
# SCORE_ONLY_NEG = np.array([1, 2, 3, 4, 5], dtype=str)
SCORE_ONLY_POS = np.array(['(5).', '(4).', '(3).', '(2).', '(1).'], dtype=str)
SCORE_ONLY_NEG = np.array(['(1).', '(2).', '(3).', '(4).', '(5).'], dtype=str)
SCORE_DESC_POS = np.array(reduce(
    lambda lst, z: lst + [z[0] + " " + z[1]], zip(SCORE_ONLY_POS, DESC_ONLY), []))
SCORE_DESC_NEG = np.array(reduce(
    lambda lst, z: lst + [z[0] + " " + z[1]], zip(SCORE_ONLY_NEG, DESC_ONLY), []))


#####  TEMPLATE DICTIONARY  #####
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


#####  SYMMETRY EXPERIMENT ORDER  #####
ORIGINAL_ORDER = [0, 1, 2, 3, 4]
REVERSE_ORDER = [4, 3, 2, 1, 0]
ORDER_I = [1, 0, 4, 3, 2]
ORDER_II = [4, 0, 2, 3, 1]
ORDER_III = [3, 1, 4, 0, 2]
ORDERS = {
    'original': ORIGINAL_ORDER,
    'reverse': REVERSE_ORDER,
    'order-I': ORDER_I,
    'order-II': ORDER_II,
    'order-III': ORDER_III
}