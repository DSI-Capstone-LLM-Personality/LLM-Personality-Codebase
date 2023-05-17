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
import argparse
import yaml
from util.utils import *
from MPI.mpi import *
from model.language_model import *
from template.templates import *
from util.human_ans_parser import get_item_key_map


SCORE_TO_CHOICE = {
    '+': {
        1: 'VA',
        2: 'MA',
        3: 'NANI',
        4: 'MI',
        5: 'VI'
    },
    '-': {
        5: 'VA',
        4: 'MA',
        3: 'NANI',
        2: 'MI',
        1: 'VI'
    }
}


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='configuration file')
args = parser.parse_args()
assert args.config is not None, 'Please specify the config .yml file to proceed.'
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


# GET CONFIGURATION
dset = config['dataset']['dset']
regime, category = config['experiment'].values()
model_config = config['model']
family, version = model_config['family'], model_config['version']

if family in ['GPTNEO', 'GPTNEOX', 'BART', 'FLAN-T5', 'T0', 'OPT']:
    version = version.split('/')[1]

prompt_template = config['template']['prompt']
# ans_type = config['template']['ans_type']
description = config['template']['description']
is_lower = config['template']['is_lower_case']

# MANUALLY CHANGE THIS
# ans_type = 'index-desc'
ans_type='index'
# ans_type='desc'

mpis_dir = f"checkpoint/mpis/{regime}/{category}/{version}/{description}/{ans_type}/"
filename = log_fname(dset, model_config, description)

tmp_name = prompt_template
if is_lower:
    tmp_name = tmp_name.replace('og', 'lc')
filename += f"_{tmp_name}"
fname = f"{mpis_dir}{filename}_[original]"

if ans_type is not None and regime == "Constraint":
    fname += f"_[{ans_type}]"

def calculate_scores(llm_obs, human_obs, disable_display=False):
    return np.array([wasserstein_distance(llm_obs, obs) for obs in tqdm(human_obs, disable=disable_display)])
def dist_to_obs(dist):
    pass
def obs_to_dist(obs):
    dist = []
    for x in tqdm(obs):
        counter = Counter(x)
        dist.append([counter[i] for i in range(1, 6, 1)])
    return np.array(dist)

 # ----------------------------- READ Dataset ----------------------------- #
IPIP120_df = pd.read_csv("Dataset/Human Data/IPIP120.csv")
n_rows = IPIP120_df.shape[0]

qt_df = pd.read_excel('Dataset/Human Data/IPIP-NEO-ItemKey.xls')
item_key_map = get_item_key_map(qt_df, int(120))
# IPIP120_df.head()


# ----------------------------- Language Model Output ----------------------------- #
MODEL = version.upper()
print(MODEL)
os.makedirs(f"human/{MODEL}/{description}/", exist_ok=True)
log = f"human/{MODEL}/{description}/log.txt"
original_stdout = sys.stdout
with open(log, 'w') as f:
    sys.stdout = f
    ckpt = torch.load(f"{fname}.pt",map_location=DEVICE)

    dset_120_path = "Dataset/ocean_120.csv"  
    df_120 = read_mpi(dset_120_path)
    # print(ckpt.text)
    text, label, raw = np.array(ckpt.text), np.array(ckpt.label), np.array(ckpt.mpi_df['label_raw'])
    label_120, text_120, raw_120, key_120 = df_120['label_ocean'], df_120['text'], df_120['label_raw'], df_120['key']
    key = np.array(ckpt.plus_minus)
    # ic(key)
    mask = []
    selected_keys = []
    for idx, (x, y, z) in enumerate(zip(label, text, raw)):
        for lbl, q, r in zip(label_120, text_120, raw_120):
            if x.strip() == lbl.strip() and y.strip() == q.strip() and z.strip() == r.strip():
                mask.append(idx)
                selected_keys.append(key[idx])
                break

    selected_keys = np.array(selected_keys)
    # ic(selected_keys.shape)
    mask = np.array(mask)

    # ic(mask.shape)
    LLM_OBS = defaultdict(list)
    for idx in mask:
        trait = ckpt.label[idx]
        score = ckpt.scores[idx]
        LLM_OBS[trait].append(score)
    for key in LLM_OBS:
        LLM_OBS[key] = np.array(LLM_OBS[key])
    print(LLM_OBS)
    # # For test
    # LLM_OBS = {
    #     'O': np.array([1, 5] * 12),
    #     'C': np.array([1]* 11 + [5]*13),
    #     'E': np.array([1]* 18 + [5]*6),
    #     'A': np.array([1]* 7 + [5]*17),
    #     'N': np.array([1]* 17 + [5]*7)
    # }
    # ----------------------------- Raw Observations ----------------------------- #
    # Observation
    # OBS = {}
    # for trait in "OCEAN":
    #     coi = list(filterfalse(lambda k: item_key_map[k][1] != trait, item_key_map))
    #     OBS[trait] = np.array(IPIP120_df[coi])
    # torch.save(OBS, "human/HUMAN_OBS.pt")
    OBS = torch.load("human/HUMAN_OBS.pt")
    # DISTRIBUTION
    HUMAN_CHOICE_OBS = {}
    for trait in tqdm(['O', 'C', 'E', 'A', 'N']):
        coi = list(filterfalse(lambda k: item_key_map[k][1] != trait, item_key_map))
        data = np.array(IPIP120_df[coi])
        # Map scores to choice
        for i in range(data.shape[0]):
            if 0 in data:
                continue
            for j in range(data.shape[1]):
                data[i][j] = SCORE_TO_CHOICE[item_key_map[coi[i]][0]]  

        HUMAN_CHOICE_OBS[trait] = data
    ic(HUMAN_CHOICE_OBS['O'].shape)
    # ----------------------------- Wasserstein Distance Calculation ----------------------------- #
    # LLM scores
    # LLM_SCORES = {}
    # for trait in 'OCEAN':
    #     LLM_SCORES[trait] = calculate_scores(LLM_OBS[trait], OBS[trait])
    # HUMAN estimation
    OBS_SCORES = torch.load("human/HUMAN_OBS_SCORES.pt")

    # ----------------------------- Wasserstein Distance Plot (HUMAN vs LLM) ----------------------------- #
    def plot_distribution(dist1, dist2, c):
        plt.hist(dist1, bins=c['num_bins'], density=True, alpha=c['alpha'], color=c['c1'], label=c['l1'])
        plt.hist(dist2, bins=c['num_bins'], density=True, alpha=c['alpha'], color=c['c2'], label=c['l2'])
        sns.kdeplot(dist1, linewidth=1, color=c['c1'], bw_adjust=2)
        sns.kdeplot(dist2, linewidth=1, color=c['c2'], bw_adjust=2)
        plt.legend()
        plt.xlabel("Wasserstein Distance")
        plt.ylabel("Density")
        plt.title(f"Pairwise Wasserstein Distance Distribution - Trait {c['trait']}")
        os.makedirs(f"human/{MODEL}/{description}/", exist_ok=True)
        plt.savefig(f"human/{MODEL}/{description}/{c['l1'] + '-' + c['l2']}-{c['trait']}.jpg", dpi=500)
        plt.close()

    # Configuration
    llm_human_config = {
        'num_bins': 30,
        'alpha': 0.3,
        # 'c1': '#0000a7',
        'c1': 'navy',
        # 'c2': '#eecc16',
        'c2': '#c1272d',
        # 'c1': '#b3b3b3',
        'trait': 'O',
        'l1': 'Human',
        'l2': f'{MODEL}',
        'title': f'{MODEL}-Human'
    }

    # for trait in 'OCEAN':
    #     dist1 = OBS_SCORES[trait]
    #     dist2 = LLM_SCORES[trait]
    #     llm_human_config['trait'] = trait
    #     plot_distribution(dist1, dist2, llm_human_config)

    # # ----------------------------- Wasserstein Distance Plot (LLM ONLY) ----------------------------- #
    def plot_llm_distribution(dist, c):
        plt.hist(dist, bins=c['num_bins'], density=True, alpha=c['alpha'], color=c['c2'], label=c['l2'])
        sns.kdeplot(dist, linewidth=1, color=c['c2'], bw_adjust=2)
        plt.legend()
        plt.xlabel("Wasserstein Distance")
        plt.ylabel("Density")
        plt.title(f"Pairwise Wasserstein Distance Distribution - Trait {c['trait']}")
        os.makedirs(f"human/{c['l2']}/{description}/", exist_ok=True)
        plt.savefig(f"human/{c['l2']}/{description}/{c['l2']}-{c['trait']}.jpg", dpi=1200)
        plt.close()
    llm_config = {
        'num_bins': 30,
        'alpha': 0.3,
        # 'c1': '#0000a7',
        'c1': 'navy',
        # 'c2': '#eecc16',
        'c2': '#c1272d',
        # 'c1': '#b3b3b3',
        'trait': 'O',
        'l1': 'Human',
        'l2': f'{MODEL}',
        'title': f'{MODEL}'
    }
    # for trait in 'OCEAN':
    #     dist = LLM_SCORES[trait]
    #     llm_config['trait'] = trait
    #     plot_llm_distribution(dist, llm_config)

    # # ----------------------------- Wasserstein Distance Threshold ----------------------------- #
    def find_percentage_below(scores, threshold):
        mask = scores <= threshold
        num = sum(mask)
        p = num/ len(scores)
        return mask, num, p
    thres_lst = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    # print(thres_lst)
    # for trait in 'OCEAN':
    #     p_lst = []
    #     p_below = f"{trait}"
    #     for threshold in thres_lst:
    #         mask, num,  p = find_percentage_below(OBS_SCORES[trait], threshold)
    #         # ic(num)
    #         # ic(f"{p*100:.4f}%")
    #         p_lst.append(f"{p*100:.4f}%")
    #         p_below += f" & ${p*100:.4f}\%$"
    #     # print(trait)
    #     p_below += "\\\\"
    #     print(p_below)
        # print(p_lst)

    # ----------------------------- ENTROPY CALCULATION ----------------------------- #
    def normalize(arr): return arr / np.sum(arr, axis=1, keepdims=True)
    def entropy(arr):
        tmp = arr
        tmp[arr == 0] = 1
        log_arr = np.emath.logn(5, tmp) # 5 classes so log base 5
        return -np.sum(arr * log_arr, axis=1)

    # ----------------------------- Distribution & Entropy ----------------------------- #
    LLM_DIST = {}
    for trait in 'OCEAN':
        ic(LLM_OBS[trait])
        LLM_DIST[trait] = normalize(obs_to_dist(LLM_OBS[trait].reshape((1, -1))))
    LLM_ENTROPY = {}
    for trait in 'OCEAN':
        ic(LLM_DIST[trait])
        LLM_ENTROPY[trait] = entropy(LLM_DIST[trait])
        ic(LLM_ENTROPY[trait])

    # HUMAN_DIST = {}
    # for trait in 'OCEAN':
    #     HUMAN_DIST[trait] = normalize(obs_to_dist(OBS[trait]))

    # HUMAN_ENTROPY = {}
    # for trait in 'OCEAN':
    #     HUMAN_ENTROPY[trait] = entropy(HUMAN_DIST[trait])

    # torch.save(HUMAN_ENTROPY, "human/HUMAN_ENTROPY.pt")
    HUMAN_ENTROPY = torch.load("human/HUMAN_ENTROPY.pt")
    # ----------------------------- ENTROPY PLOT ----------------------------- #
    def plot_entropy(dist, llm_entropy, c):
        plt.hist(dist, bins=c['num_bins'], density=True, alpha=c['alpha'], color=c['c1'], label=c['l1'])
        plt.axvline(x=llm_entropy, color=c['c2'], linestyle='dashed', label=c['l2'])
        # PLOT rejection region
        rej = np.quantile(dist, 0.05)
        # plt.axvspan(0, rej, alpha=0.4, color='red')
        ax = sns.kdeplot(dist, bw_adjust=2)
        # Below code to shade partial region is from 
        # https://stackoverflow.com/a/49100655

        # Get all the lines used to draw the density curve 
        kde_lines = ax.get_lines()[-1]
        kde_x, kde_y = kde_lines.get_data()

        # Use Numpy mask to filter the lines for region 
        # reresenting height greater than 60 inches 
        mask = kde_x < rej
        filled_x, filled_y = kde_x[mask], kde_y[mask]

        # Shade the partial region 
        ax.fill_between(filled_x, y1=filled_y, color='#c1272d', alpha=0.5)

        sns.kdeplot(dist, linewidth=1, color=c['c1'], bw_adjust=2)
        plt.legend()
        plt.xlabel("Entropy")
        plt.ylabel("Density")

        plt.annotate(f'{round(llm_entropy.item(),2)}', xy=(0.25, 1), xytext=(llm_entropy.item()+0.05, 1.0))
        plt.xticks([0, 0.2, 0.8, 1, rej.item()], [0, 0.2, 0.8, 1, round(rej.item(),2)])
        plt.title(f"Entropy Distribution - Trait {c['trait']}")
        os.makedirs(f"human/{MODEL}/{description}/", exist_ok=True)
        plt.savefig(f"human/{MODEL}/{description}/Entropy-{c['l1'] + '-' + c['l2']}-{c['trait']}.jpg", dpi=1000)
        plt.close()

    def plot_mean_std(dist, llm_stats, c, statistic):
        plt.hist(dist, bins=c['num_bins'], density=True, alpha=c['alpha'], color=c['c1'], label=c['l1'])
        plt.axvline(x=llm_stats, color=c['c2'], linestyle='dashed', label=c['l2'])
        # PLOT rejection region
        lb, ub = np.quantile(dist, 0.005), np.quantile(dist, 0.995)
        # plt.axvspan(0, rej, alpha=0.4, color='red')
        ax = sns.kdeplot(dist, bw_adjust=2)
        # Below code to shade partial region is from 
        # https://stackoverflow.com/a/49100655

        # Get all the lines used to draw the density curve 
        kde_lines = ax.get_lines()[-1]
        kde_x, kde_y = kde_lines.get_data()
        # left tail
        mask = kde_x < lb
        filled_x, filled_y = kde_x[mask], kde_y[mask]
        ax.fill_between(filled_x, y1=filled_y, color='#c1272d', alpha=0.5)
        # right tail
        mask = kde_x > ub
        filled_x, filled_y = kde_x[mask], kde_y[mask]
        ax.fill_between(filled_x, y1=filled_y, color='#c1272d', alpha=0.5)

        sns.kdeplot(dist, linewidth=1, color=c['c1'], bw_adjust=2)
        plt.legend()
        plt.xlabel(f"{statistic}")
        plt.ylabel("Density")

        if statistic == 'Mean':
            plt.annotate(f'{round(llm_stats,2)}', xy=(0.25, 1), xytext=(llm_stats+0.05, 0.5))
            plt.xticks([1, lb, ub, 5], [1, round(lb, 2), round(ub, 2), 5])
        else:
            plt.annotate(f'{round(llm_stats,2)}', xy=(0.25, 1), xytext=(llm_stats+0.05, 0.8))
            plt.xticks([0, lb, ub, 2], [0, round(lb, 2), round(ub, 2), 2])
        plt.title(f"{statistic} Distribution - Trait {c['trait']}")
        os.makedirs(f"human/{MODEL}/{description}/", exist_ok=True)
        plt.savefig(f"human/{MODEL}/{description}/{statistic}-{c['l1'] + '-' + c['l2']}-{c['trait']}.jpg", dpi=1000)
        plt.close()

    entropy_config = {
        'num_bins': 30,
        'alpha': 0.3,
        'c1': 'navy',
        'c2': 'black',
        # 'c2': '#c1272d',
        'trait': 'O',
        'l1': 'Human',
        'l2': f'{MODEL}',
        'title': f'{MODEL}-Human'
    }

    # Load checkpoints
    HUMAN_MEAN = torch.load('human/HUMAN_MEAN.pt')
    HUMAN_STD = torch.load('human/HUMAN_STD.pt')

    print('ENTROPY')
    for trait in 'OCEAN':
        print(f"{LLM_ENTROPY[trait]}")
    print('MEAN')
    for trait in 'OCEAN':
        print(f"{np.mean(LLM_OBS[trait]).item()}")
    print('STD')
    for trait in 'OCEAN':
        print(f"{np.std(LLM_OBS[trait]).item()}")

    for trait in 'OCEAN':
        dist = HUMAN_ENTROPY[trait]
        llm_entropy = LLM_ENTROPY[trait]
        entropy_config['trait'] = trait
        plot_entropy(dist, llm_entropy, entropy_config)
        plot_mean_std(HUMAN_MEAN[trait], np.mean(LLM_OBS[trait]).item(), entropy_config, 'Mean')
        plot_mean_std(HUMAN_STD[trait], np.std(LLM_OBS[trait]).item(), entropy_config, 'STD')

        probs = f"\\textbf{{{trait}}}"
        # Compute quantile
        dist = np.array(dist)
        quantile = 100 * sum(dist <= llm_entropy) / len(dist)
        # print(f"{quantile:.4f}\%")
        probs += f" & ${quantile:.4f}\%$"
        # Compute probability
        p = 100 * sum(dist == llm_entropy) / len(dist)
        # print(f"{p:.4f}%")
        probs += f" & ${p:.4f}\%$ "
        
        for eps in [0.0001, 0.001, 0.01]:
            mask1 = (dist > (llm_entropy-eps))
            mask2 = (dist < (llm_entropy+eps))
            mask = mask1 & mask2
            prob = 100 * sum(mask)/ len(dist)
            probs += f" & ${prob:.3f}\%$"
        probs += "\\\\"
        print(probs)
    f.close()
    sys.stdout = original_stdout