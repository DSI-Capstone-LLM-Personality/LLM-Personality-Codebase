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
