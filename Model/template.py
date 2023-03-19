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
LETTER_INDEX = np.array(['(A).', '(B).', '(C).', '(D).', '(E).'])
NUMBER_INDEX = np.array(['(5).', '(4).', '(3).', '(2).', '(1).'])
# Reverse
NUMBER_INDEX_REV = np.array(['(1).', '(2).', '(3).', '(4).', '(5).'])

INDEX = {
    'letter': LETTER_INDEX,
    'number': NUMBER_INDEX,
    'number-rev': NUMBER_INDEX_REV  # deprecated!
}
#####  DESC TEMPLATE #####
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
