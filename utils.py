import torch
import numpy as np
from functools import reduce


# OCEAN BASICS
OCEAN = ['O', 'C', 'E', 'A', 'N']
# ------------------------------- #
# MPI IDX-ANSWER-SCORE CONVERSION #
# ------------------------------- #
# TODO: (Xiaoyang) Wrap all of these into a .yml file later.

MPI_NUM_CHOICES = 5

# PROMPT TEMPLATE
MPI_PROMPT = "Please choose from the following options to identify how accurately this statement describes you."
PROMPT_TEMPLATE = {'mpi-style': MPI_PROMPT}
# Letter Answer Template
LETTER_ONLY = np.array(['(A).', '(B).', '(C).', '(D).', '(E).'])
DESC_ONLY = np.array([
    "Very Accurate",
    "Moderately Accurate",
    "Neither Accurate Nor Inaccurate",
    "Moderately Inaccurate",
    "Very Inaccurate"])
LETTER_DESC = np.array(reduce(
    lambda lst, z: lst + [z[0] + " " + z[1]], zip(LETTER_ONLY, DESC_ONLY), []))
# ANSWER TEMPLATE
ANSWER_TEMPLATE = {
    'letter-only': LETTER_ONLY,
    'desc-only': DESC_ONLY,
    'letter-desc': LETTER_DESC
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
# [dset]_[model | version]_[choice_type]_[ll_type]
def log_fname(dset, model_desc, answer_type, ll_type=None):
    family, version = model_desc['family'], model_desc['version']
    if ll_type is None:
        return f"[{dset}]_[{family}|{version}]_[{answer_type}]"
    return f"[{dset}]_[{family} | {version}]_[{answer_type}]_[{ll_type}]"


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
# choice_lst = np.array(DESC_ONLY)
# ordered_lst = shuffle_choice(choice_lst)
# option = ordered_lst_to_str(ordered_lst)
# print(ordered_lst)
# print(option)


# QF = QuestionFormatter(MPI_PROMPT, option)
# print(QF("You worry about things"))
# line()
