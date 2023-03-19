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


######  MPI CONSTANTS  ######
MPI_NUM_CHOICES = 5
######  PROMPTS  ######
MPI_PROMPT = "Please choose from the following options to identify how accurately this statement describes you."
PROMPT_TEMPLATE = {'mpi-style': MPI_PROMPT}

######  INDEXES  ######
LETTER_INDEX = np.array(['(A).', '(B).', '(C).', '(D).', '(E).'])
NUMBER_INDEX_SEM = np.array(['(5).', '(4).', '(3).', '(2).', '(1).'])
NUMBER_INDEX_SYN = np.array(['(1).', '(2).', '(3).', '(4).', '(5).'])

INDEX = {
    'letter': LETTER_INDEX,
    'number-sem': NUMBER_INDEX_SEM,
    'number-syn': NUMBER_INDEX_SYN
}
######  DESCRIPTIONS  ######
MPI_DESC = np.array([
    "Very Accurate",
    "Moderately Accurate",
    "Neither Accurate Nor Inaccurate",
    "Moderately Inaccurate",
    "Very Inaccurate"])
OTHER_DESC = np.array(['Always', 'Often', 'Sometimes', 'Rarely', 'Never'])
DESC = {
    'mpi': MPI_DESC,
    'other': OTHER_DESC
}
# ------------------------------- #
# MPI IDX-ANSWER-SCORE CONVERSION #
# ------------------------------- #
MPI_IDX_TO_SCORE_NEG = np.arange(1, 6, 1)
MPI_IDX_TO_SCORE_POS = np.arange(5, 0, -1)
MPI_SCORE = {
    "+": MPI_IDX_TO_SCORE_POS,
    "-": MPI_IDX_TO_SCORE_NEG
}


######  SYMMETRY EXPERIMENT ORDER  ######
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
