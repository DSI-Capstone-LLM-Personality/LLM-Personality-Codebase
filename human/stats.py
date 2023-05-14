import torch

from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from icecream import ic
from collections import Counter
from itertools import filterfalse
import numpy as np
import seaborn as sns
from itertools import filterfalse
# sys.path.append("../")

from util.utils import *
from MPI.mpi import *
from model.language_model import *
from template.templates import *
from util.human_ans_parser import get_item_key_map

device = torch.device('cpu')
# pt_path = '/Users/tree/Desktop/Capstone/LLM-Personality-Codebase-main/checkpoint/mpis/Constraint/order-symmetry/opt-13b/non-index/desc/'
# pt_file = 'checkpoint/mpis/Constraint/order-symmetry/opt-13b/non-index/desc/[ocean_988]_[OPT|opt-13b]_[non-index]_[lc]-[ns]-[type-i]-[ans-i]_[order-I]_[desc].pt'
# # pt_file = 'checkpoint/mpis/Early Results/GPT2-Base/order/[ocean_120]_[GPT2|gpt2]_[non-index]_[order-I].pt'
# dset_120_path = "Dataset/ocean_120.csv"
# ckpt = torch.load(pt_file, map_location=device)     
# df_120 = read_mpi(dset_120_path)
# # print(ckpt.text)
# text = np.array(ckpt.text)
# ic(text.shape)
# label = np.array(ckpt.label)
# ic(label.shape)
# raw = np.array(ckpt.mpi_df['label_raw'])
# label_120, text_120, raw_120 = df_120['label_ocean'], df_120['text'], df_120['label_raw']
# mask = []
# for x, y, z in zip(label, text, raw):
#     match=False
#     for lbl, q, r in zip(label_120, text_120, raw_120):
#         if x.strip() == lbl.strip() and y.strip() == q.strip() and z.strip() == r.strip():
#             match = True
#             break
#     mask.append(match)

# mask = np.array(mask)
# ic(sum(mask))
# ic(mask.shape)
# ic(text[mask].shape)
# ic(np.setdiff1d(df_120['text'], list(text[mask])))
# ic(np.setdiff1d(df_120['text'], list(text[mask])).shape)
# PROCESSING



IPIP120_df = pd.read_csv("Dataset/Human Data/IPIP120.csv")
n_rows = IPIP120_df.shape[0]

qt_df = pd.read_excel('Dataset/Human Data/IPIP-NEO-ItemKey.xls')
item_key_map = get_item_key_map(qt_df, int(120))
# IPIP120_df.head()

LLM_OBS = {
    'O': np.array([1, 5] * 12),
    'C': np.array([1]* 11 + [5]*13),
    'E': np.array([1]* 18 + [5]*6),
    'A': np.array([1]* 7 + [5]*17),
    'N': np.array([1]* 17 + [5]*7)
}

# Observation
OBS = {}

for trait in "OCEAN":
    coi = list(filterfalse(lambda k: item_key_map[k][1] != trait, item_key_map))
    OBS[trait] = np.array(IPIP120_df[coi])


def calculate_scores(llm_obs, human_obs, disable_display=False):
    return np.array([wasserstein_distance(llm_obs, obs) for obs in tqdm(human_obs, disable=disable_display)])

LLM_SCORES = {}
for trait in 'OCEAN':
    LLM_SCORES[trait] = calculate_scores(LLM_OBS[trait], OBS[trait])


def dist_to_obs(dist):
    pass


def obs_to_dist(obs):
    dist = []
    for x in tqdm(obs):
        counter = Counter(x)
        dist.append([counter[i] for i in range(1, 6, 1)])
    return np.array(dist)


OBS_SCORES = torch.load("human/HUMAN_OBS_SCORES.pt")
config = {
    'num_bins': 30,
    'alpha': 0.3,
    # 'c1': '#0000a7',
    'c1': 'navy',
    # 'c2': '#eecc16',

    'c2': '#c1272d',
    # 'c1': '#b3b3b3',
    'trait': 'O',
    'l1': 'Human',
    # 'l2': 'OPT-125M',
    'l2': 'Test',
    # 'l2': 'Human Test',
    # 'l2': 'OPT-13B',
    'title': 'OPT-125M-Human'
}


def plot_distribution(dist1, dist2, c):
    plt.hist(dist1, bins=c['num_bins'], density=True, alpha=c['alpha'], color=c['c1'], label=c['l1'])
    plt.hist(dist2, bins=c['num_bins'], density=True, alpha=c['alpha'], color=c['c2'], label=c['l2'])
    sns.kdeplot(dist1, linewidth=1, color=c['c1'], bw_adjust=2)
    sns.kdeplot(dist2, linewidth=1, color=c['c2'], bw_adjust=2)
    plt.legend()
    plt.xlabel("Wasserstein Distance")
    plt.ylabel("Density")
    plt.title(f"Pairwise Wasserstein Distance Distribution - Trait {c['trait']}")
    plt.savefig(f"human/{c['l1'] + '-' + c['l2']}-{c['trait']}.jpg", dpi=1200)
    plt.close()

for trait in 'OCEAN':
    dist1 = OBS_SCORES[trait]
    # dist2 = HUMAN_VAL_SCORES[trait].reshape((-1,))
    dist2 = LLM_SCORES[trait]
    config['trait'] = trait
    plot_distribution(dist1, dist2, config)

def plot_llm_distribution(dist, c):
    plt.hist(dist, bins=c['num_bins'], density=True, alpha=c['alpha'], color=c['c2'], label=c['l2'])
    sns.kdeplot(dist, linewidth=1, color=c['c2'], bw_adjust=2)
    plt.legend()
    plt.xlabel("Wasserstein Distance")
    plt.ylabel("Density")
    plt.title(f"Pairwise Wasserstein Distance Distribution - Trait {c['trait']}")
    plt.savefig(f"human/{c['l2']}-{c['trait']}.jpg", dpi=1200)
    plt.close()

for trait in 'OCEAN':
    dist = LLM_SCORES[trait]
    config['trait'] = trait
    plot_llm_distribution(dist, config)


def find_percentage_below(scores, threshold):
    mask = scores <= threshold
    num = sum(mask)
    p = num/ len(scores)
    return mask, num, p
for trait in 'OCEAN':
    p_lst = []
    for threshold in [1, 1e-1, 1e-2, 1e-3, 1e-4]:
        mask, num,  p = find_percentage_below(OBS_SCORES[trait], threshold)
        # ic(num)
        # ic(f"{p*100:.4f}%")
        p_lst.append(f"{p*100:.4f}%")
    print(trait)
    print(p_lst)


def normalize(arr): return arr / np.sum(arr, axis=1, keepdims=True)
def entropy(arr):
    tmp = arr
    tmp[arr == 0] = 1
    log_arr = np.emath.logn(5, tmp) # 5 classes so log base 5
    return -np.sum(arr * log_arr, axis=1)

LLM_DIST = {}

for trait in 'OCEAN':
    LLM_DIST[trait] = normalize(obs_to_dist(LLM_OBS[trait].reshape((1, -1))))

LLM_ENTROPY = {}
for trait in 'OCEAN':
    LLM_ENTROPY[trait] = entropy(LLM_DIST[trait])

HUMAN_DIST = {}
for trait in 'OCEAN':
    HUMAN_DIST[trait] = normalize(obs_to_dist(OBS[trait]))

HUMAN_ENTROPY = {}
for trait in 'OCEAN':
    HUMAN_ENTROPY[trait] = entropy(HUMAN_DIST[trait])

config = {
    'num_bins': 30,
    'alpha': 0.3,
    'c1': 'navy',
    # 'c2': '#eecc16',
    'c2': '#c1272d',
    'trait': 'O',
    'l1': 'Human',
    'l2': 'OPT-125M',
    'title': 'OPT-125M-Human'
}


def plot_entropy(dist, llm_entropy, c):
    plt.hist(dist, bins=c['num_bins'], density=True, alpha=c['alpha'], color=c['c1'], label=c['l1'])
    plt.axvline(x=llm_entropy, color=c['c2'], linestyle='dashed', label=c['l2'])
    sns.kdeplot(dist, linewidth=1, color=c['c1'], bw_adjust=2)
    plt.legend()
    plt.xlabel("Entropy")
    plt.ylabel("Density")
    plt.title(f"Entropy Distribution - Trait {c['trait']}")
    plt.savefig(f"human/Entropy-{c['l1'] + '-' + c['l2']}-{c['trait']}.jpg", dpi=1000)
    plt.close()

for trait in 'OCEAN':
    dist = HUMAN_ENTROPY[trait]
    llm_entropy = LLM_ENTROPY[trait]
    config['trait'] = trait
    plot_entropy(dist, llm_entropy, config)