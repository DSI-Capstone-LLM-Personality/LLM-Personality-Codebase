import torch

# ------------------------------- #
# MPI IDX-ANSWER-SCORE CONVERSION #
# ------------------------------- #
MPI_NUM_CHOICES = 5
MPI_IDX_TO_KEY = ['A', 'B', 'C', 'D', 'E']
def MPI_IDX_TO_SCORE(idx): return idx+1


# TODO: MBTI? or other personality test
