from secrets import choice
from sklearn.utils import shuffle
import torch
import numpy as np
from functools import reduce


# OCEAN BASICS
OCEAN = ['O', 'C', 'E', 'A', 'N']
# ------------------------------- #
# MPI IDX-ANSWER-SCORE CONVERSION #
# ------------------------------- #
# TODO: (Xiaoyang) Wrap all of these into a .yml file later.
MPI_PROMPT = "Please choose from the following options to identify how accurately this statement describes you."
MPI_NUM_CHOICES = 5
MPI_CHOICES_NAIVE = np.array(['(A).', '(B).', '(C).', '(D).', '(E).'])
MPI_CHOICES_DESC = np.array([
    "Very Accurate",
    "Moderately Accurate",
    "Neither Accurate Nor Inaccurate",
    "Moderately Inaccurate",
    "Very Inaccurate"])
MPI_CHOICES_ALL = np.array(reduce(
    lambda lst, z: lst + [z[0] + " " + z[1]], zip(MPI_CHOICES_NAIVE, MPI_CHOICES_DESC), []))
# CHOICE DICTIONARY (optional)
CHOICE = {
    'letter-only': MPI_CHOICES_NAIVE,
    'desc-only': MPI_CHOICES_DESC,
    'letter-desc': MPI_CHOICES_ALL
}
# SCORE DICTIONARY
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


# CHECKPOINT FILENAME FORMAT
# [dset]_[model=version]_[choice_type]_[ll_type]

def log_fname(dset, model_desc, choice_type, ll_type):
    family, version = model_desc['family'], model_desc['version']
    return f"[{dset}]_[{family}|{version}]_[{choice_type}]_[{ll_type}]"


def shuffle_choice(choice_lst):
    assert type(
        choice_lst) == np.ndarray, 'Please format your choice list as numpy array.'
    assert len(np.shape(choice_lst)) == 1
    assert len(choice_lst) > 1
    n = len(choice_lst)
    rand_idx = np.random.choice(n, n, replace=False)
    return choice_lst[rand_idx], rand_idx


def ordered_lst_to_str(ordered_lst, style='mpi'):
    if style == 'mpi':
        option = "\nOptions: "
        for choice in ordered_lst:
            option += f"\n{choice} "  # Prepend \n as designed choice
        return option + "\nAnswers: "
    else:
        assert False, 'Unrecognized option style.'


# TODO: (Xiaoyang) Finish this wrapper class
class MPIQuestionFormatter():
    def __init__(self, prompt: str, options: dict):
        self.prompt = prompt
        self.option = options

    def __call__(self, statement, key):
        question = f"Given a statement of you: \"You {statement}.\" "
        return question + self.prompt + ordered_lst_to_str(self.option[key])


def line(n=40): print("-"*n)

# Simple testing code
# choice_lst = np.array(MPI_CHOICES_DESC)
# ordered_lst = shuffle_choice(choice_lst)
# option = ordered_lst_to_str(ordered_lst)
# print(ordered_lst)
# print(option)


# QF = QuestionFormatter(MPI_PROMPT, option)
# print(QF("You worry about things"))
# line()
