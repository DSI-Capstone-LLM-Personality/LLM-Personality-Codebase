import torch

import os
import argparse
import yaml
import sys
# sys.path.append("../")

from util.utils import *
from MPI.mpi import *
from model.language_model import *
from template.templates import *

device = torch.device('cpu')
# pt_path = '/Users/tree/Desktop/Capstone/LLM-Personality-Codebase-main/checkpoint/mpis/Constraint/order-symmetry/opt-13b/non-index/desc/'
pt_file = 'checkpoint/mpis/Constraint/order-symmetry/opt-13b/non-index/desc/[ocean_988]_[OPT|opt-13b]_[non-index]_[lc]-[ns]-[type-i]-[ans-i]_[order-I]_[desc].pt'
# pt_file = 'checkpoint/mpis/Early Results/GPT2-Base/order/[ocean_120]_[GPT2|gpt2]_[non-index]_[order-I].pt'
dset_120_path = "Dataset/ocean_120.csv"
ckpt = torch.load(pt_file, map_location=device)     
df_120 = read_mpi(dset_120_path)
# print(ckpt.text)
text = np.array(ckpt.text)
ic(text.shape)
label = np.array(ckpt.label)
ic(label.shape)
raw = np.array(ckpt.mpi_df['label_raw'])
label_120, text_120, raw_120 = df_120['label_ocean'], df_120['text'], df_120['label_raw']
mask = []
for x, y, z in zip(label, text, raw):
    match=False
    for lbl, q, r in zip(label_120, text_120, raw_120):
        if x.strip() == lbl.strip() and y.strip() == q.strip() and z.strip() == r.strip():
            match = True
            break
    mask.append(match)

mask = np.array(mask)
ic(sum(mask))
ic(mask.shape)
ic(text[mask].shape)
ic(np.setdiff1d(df_120['text'], list(text[mask])))
ic(np.setdiff1d(df_120['text'], list(text[mask])).shape)