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
MPI_CHOICES = ['(A)', '(B)', '(C)', '(D)', '(E)']
MPI_CHOICES_DESC = [
    "Very Accurate",
    "Moderately Accurate",
    "Neither Accurate Nor Inaccurate",
    "Moderately Inaccurate",
    "Very Inaccurate"]
MPI_CHOICE_ALL = reduce(
    lambda lst, z: lst + [z[0] + " " + z[1]], zip(MPI_CHOICES, MPI_CHOICES_DESC), [])
MPI_IDX_TO_KEY = ['A', 'B', 'C', 'D', 'E']  # DEPRECATED
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
    def __init__(self, prompt: str, options: str, style='mpi'):
        # TODO: (Xiaoyang) this might not work for questionaire with different structures...
        # assert len(prompt) == 2, \
        #     'The template must contain prompts for both \"BEFORE\" and \"AFTER\" the \"STATEMENT\".'
        self.prompt = prompt
        self.option = options
        self.style = style

    def __call__(self, statement):
        if self.style == 'mpi':
            question = f"Given a statement of you: \"You {statement}.\" "
            return question + self.prompt + self.option
        else:
            assert False, 'Unrecognized formatting style.'


def line(n=40): print("-"*n)

# Simple testing code
# choice_lst = np.array(MPI_CHOICES_DESC)
# ordered_lst = shuffle_choice(choice_lst)
# option = ordered_lst_to_str(ordered_lst)
# print(ordered_lst)
# print(option)


# QF = QuestionFormatter(MPI_PROMPT, option)
# print(QF("You worry about things"))
line()
