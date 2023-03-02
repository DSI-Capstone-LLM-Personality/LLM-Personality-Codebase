from secrets import choice
from sklearn.utils import shuffle
import torch
import numpy as np

# ------------------------------- #
# MPI IDX-ANSWER-SCORE CONVERSION #
# ------------------------------- #
MPI_NUM_CHOICES = 5
MPI_IDX_TO_KEY = ['A', 'B', 'C', 'D', 'E']  # DEPRECATED
MPI_IDX_TO_SCORE_NEG = np.arange(1, 6, 1)
MPI_IDX_TO_SCORE_POS = np.arange(5, 0, -1)
MPI_SCORE = {
    "+": MPI_IDX_TO_SCORE_POS,
    "-": MPI_IDX_TO_SCORE_NEG
}

# print(MPI_IDX_TO_KEY)
# print(MPI_IDX_TO_SCORE_POS)
# print(MPI_IDX_TO_SCORE_NEG)


# TODO: MBTI? or other personality test
# -------------------------------- #
# MBTI IDX-ANSWER-SCORE CONVERSION #
# -------------------------------- #
# ...


# ----------------- #
# UTILITY FUNCTIONS #
# ----------------- #

def shuffle_choice(choice_lst):
    assert type(
        choice_lst) == np.ndarray, 'Please format your choice list as numpy array.'
    assert len(np.shape(choice_lst)) == 1
    assert len(choice_lst) > 1
    n = len(choice_lst)
    rand_idx = np.random.choice(n, n, replace=False)
    return choice_lst[rand_idx]


def ordered_lst_to_str(ordered_lst, style='mpi'):
    if style == 'mpi':
        option = "\nOptions: "
        for choice in ordered_lst:
            option += f"\n{choice}"  # Prepend \n as designed choice
        return option + "\nAnswer: "
    else:
        assert False, 'Unrecognized option style.'


# TODO: (Xiaoyang) Finish this wrapper class
class QuestionFormatter():
    def __init__(self, prompt, options):
        pass

    def __call__(self, statement):
        pass


# Simple testing code
choice_lst = np.array(MPI_IDX_TO_KEY)
ordered_lst = shuffle_choice(choice_lst)
option = ordered_lst_to_str(ordered_lst)
print(ordered_lst)
print(option)
