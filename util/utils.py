import openai
import torch
import numpy as np
from functools import reduce
import random
import json
from icecream import ic
from itertools import permutations
from collections import defaultdict, Counter
import difflib as dl
from template.templates import *
import os

#####  DEVICE CONFIGURATION  #####
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


######  OCEAN BASICS  ######
OCEAN = ['O', 'C', 'E', 'A', 'N']
# TODO: MBTI? or other personality test

# ----------------------------------------------------- #
#                  UTILITY FUNCTIONS                    #
# ----------------------------------------------------- #
######  CHECKPOINT FILENAME FORMAT  ######
# [dset]_[model | version]_[choice_type]_[ll_type]


def log_fname(dset, model_desc, answer_type, ll_type=None):
    family, version = model_desc['family'], model_desc['version']
    if family in ['GPTNEO', 'GPTNEOX', 'BART', 'T0', 'OPT']:
        version = version.split('/')[1]
    if ll_type is None:
        return f"[{dset}]_[{family}|{version}]_[{answer_type}]"
    return f"[{dset}]_[{family} | {version}]_[{answer_type}]_[{ll_type}]"


######  RANDOM ORDER SELECTION -- SYMMETRY EXPERIMENT  ######
def shuffle_choice(choice_lst, given_idx=None):
    assert type(
        choice_lst) == np.ndarray, 'Please format your choice list as numpy array.'
    assert len(np.shape(choice_lst)) == 1
    assert len(choice_lst) > 1
    if given_idx is not None:
        return choice_lst[given_idx], given_idx
    n = len(choice_lst)
    rand_idx = np.random.choice(n, n, replace=False)
    return choice_lst[rand_idx], rand_idx


def order_distance(idx):
    lst1 = np.array([idx[0]] + list(idx))
    lst2 = np.array(list(idx) + [idx[-1]])
    return np.sum(np.abs(lst1 - lst2))


def permute_orders(n):
    return np.array(list(permutations(np.arange(n))))


def order_ranking_dict(permutation, verbose=False):
    orders = defaultdict(list)
    for order in permutation:
        orders[order_distance(order)].append(order)
    if verbose:
        for key in orders:
            print(f"{key:<2} | {len(orders[key])}")
    return orders


# MPI_IDX_ORDERS = order_ranking_dict(permute_orders(MPI_NUM_CHOICES))


def ordered_lst_to_str(ordered_lst):
    option = ""
    for choice in ordered_lst:
        option += f"{choice}\n"  # Append \n as designed choice
    return option

# Simple test cases:
# ic(order_distance([5, 4, 3, 2, 1]))
# ic(order_distance([1, 2, 3, 4, 5]))
# ic(order_distance([1, 2, 5, 3, 4]))
# ic(permute_orders(5).shape)
# orders = order_ranking_dict(permute_orders(5), True)
# ic(np.array(orders[11])[np.random.choice(8, 1)])

# order selection result
# 4  | 2 (all)
# 5  | 4
# 6  | 14
# 7  | 32 # [1, 0, 4, 3, 2]
# 8  | 18
# 9  | 28 # [4, 0, 2, 3, 1]
# 10 | 14
# 11 | 8  # [3, 1, 4, 0, 2]

###### FORMATTERS  ######


class MPIOptionFormatter():
    def __init__(self, index, desc, is_lower_case=False):
        self.index = index
        self.desc = desc
        # ic(self.desc)
        self.is_lower_case = is_lower_case
        if self.is_lower_case:
            # print(self.desc)
            self.desc = {k: np.array([x.lower() for x in v])
                         for k, v in self.desc.items()}

    def __call__(self, order, shuffle_both=None):
        if order is not None:
            assert shuffle_both is not None
            if shuffle_both and self.index is not None:
                self.index = {k: v[order] for k, v in self.index.items()}
            # ic(self.desc)
            self.desc = {k: v[order] for k, v in self.desc.items()}
        if self.index is not None:
            return {k: concat(self.index[k], self.desc[k]) for k in ['+', '-']}
        else:
            return self.desc


def lc_dict(target):
    return {k: np.array([x.lower() for x in v])
                        for k, v in target.items()}

def MPI_options_to_answers(index, desc, option, ans_type, is_lower=False, order=None):
    if ans_type == 'desc':
        if is_lower:
            desc = lc_dict(desc)
        return desc
    elif ans_type == 'index':
        return index
    elif ans_type == 'index-desc':
        assert order is not None
        return {k: option[k][order] for k in ['+', '-']}
    else:
        assert False, 'Unrecognized answer template type.'


class MPIQuestionFormatter():
    def __init__(self, prompt: str, options: dict):
        self.prompt = prompt
        self.option = options

    def __call__(self, statement, key):
        return self.prompt.format(item=statement.lower(), options=ordered_lst_to_str(self.option[key]))


######  UTILITY FUNCTIONS  ######
def line(n=80, is_print=True):
    if is_print:
        print("-"*n)
    else:
        return "-"*n


def load_mpi_instance(filename):
    return torch.load(filename, map_location=torch.device('cpu'))


def format_ans_distribution_latex_table(mpi):
    assert mpi.answered
    ans_dist_table = []
    for sign in ['+', '-']:
        stat = Counter(mpi.preds_key[mpi.plus_minus == sign])
        for item in mpi.mpi_choice_lst[sign]:
            ans_dist_table.append(stat[item])
    # Change order if necessary
    order = np.array([[a, b] for a, b, in zip(
        np.arange(5), np.arange(5, 10, 1))]).flatten()
    # print(order)
    ans_dist_table = np.array(ans_dist_table)[order]
    # Formatting
    out = ""
    for vals in ans_dist_table:
        out += f"& ${vals}$ "
    if mpi.regime == "Open-Vocab":
        out += f"& ${(100 * mpi.processor.valid_idx.sum() / mpi.processor.n):.2f}\%$ "
    out += "\\\\"
    return out


def format_ocean_latex_table(mpi):
    # Note that the output is a single row of table
    ocean_scores = mpi.OCEAN
    ocean_table = []
    for item in OCEAN:
        vals = torch.tensor(ocean_scores[item], dtype=torch.float32)
        mean, std = torch.mean(vals).item(), torch.std(vals).item()
        ocean_table.append(mean)
        ocean_table.append(std)
    # Formatting
    out = ""
    for vals in ocean_table:
        out += f"& ${vals:.2f}$ "
    out += "\\\\"
    return out


def set_seed(seed):
    # TODO: (Xiaoyang) change this function later
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


######  READ OPENAI API KEY ######
def read_api_key(path="", identifier="xysong"):
    import os
    # print('Get current working directory : ', os.getcwd())
    with open(path + "keys.json", 'r') as f:
        keys = json.load(f)
        openai_api_keys = keys["openai-api-key"]
        assert identifier in openai_api_keys
        return openai_api_keys[identifier]
# read_api_key()

######  RESPONSE PROCESSER  ######

# TODO: (Xiaoyang & Kiyan): redo this class to make sure it works well.


class PROCESSER():
    def __init__(self, choice, method='closest-match', verbose=False):
        # TODO: (Xiaoyang): make this more generic later...
        self.keywords = SCORING_KEYWORDS
        self.choices = [x.lower() for x in choice]
        self.method = method
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.raw_responses, self.woi_lst = [], []
        self.processed_response, self.invalid_idx = [], []
        self.valid_idx, self.n = [], 0

    def __call__(self, response):
        response = response.strip()
        self.raw_responses.append(response)
        response = response.lower()
        words = response.split(" ")
        woi = []  # woi stands for "words of interests"
        for word in words:
            if word in self.keywords:
                woi.append(word)
        processed_response = ' '.join(woi)
        if self.verbose:
            print(f"PROCESSOR: {response} --> {processed_response}")
        # Store statistics
        self.processed_response.append(processed_response)
        self.woi_lst.append(woi)
        self.n += 1
        # TODO: (Team): probably one word is also probelmatic? discuss later...
        if len(woi) <= 1 or response not in self.choices:
            if len(woi) <= 1:
                self.valid_idx.append(False)
                return None, -1
            # Use this line when reproducing MPI results
            # self.valid_idx.append(True)
            # Comment this line when doing other tasks
            self.valid_idx.append(False)
            return None, -1  # -1 indicates that this response is not valid

        # Update valid mask: this is used for statistic display later...
        self.valid_idx.append(True)
        # Compute matched output
        if self.method == 'closest-match':
            match = dl.get_close_matches(
                processed_response, self.choices, n=1, cutoff=0)
            idx = self.choices.index(match[0])
            return processed_response, idx
        else:
            assert False, 'Unrecognized Processing Method.'

    def display_stats(self):
        print(f"Details about generated responses and PROCESSOR results.")
        line()
        print(f"Total number of responses given: {self.n}")
        self.valid_idx = np.array(self.valid_idx)
        print(
            f"Total number of invalid responses: {(~self.valid_idx).sum()}")
        n_correct = self.valid_idx.sum()
        print(f"Valid Percentage: {np.round(100 * (n_correct / self.n), 2)}%")
        # TODO: (Xiaoyang) Finish statistic logging
        responses = dict(Counter(self.raw_responses))
        # print(responses)
        print(f"There are {len(responses)} different answers generated!")
        for item, val in responses.items():
            print(f">> {item}\n>> Count: {val}")
            line(60)
        line()


# Test processor code
# openai.api_key = read_api_key("", 'kiyan')
# openai.api_key = read_api_key("", 'xysong')
# processor = PROCESSER(verbose=True)
# # item = "worry about things"
# item = "have difficulty imagining things"
# eg_q = MPI_TEMPLATE.format(item=item) + ordered_lst_to_str(MPI_DESC)
# response = openai.Completion.create(
#     engine="text-davinci-002", prompt=eg_q, temperature=0.1, max_tokens=10, top_p=0.95, logprobs=1)
# ic(response['choices'][0]['text'].strip())
# ic(processor(response['choices'][0]['text']))


if __name__ == '__main__':
    # Formatting latex (remove this later)
    mpi_dir = "checkpoint/mpis/"
    # TO USE: change folder name and filename
    # FOR OPEN VOCAB
    # folder = "Open-Vocab/order-symmetry/text-davinci-002/non-index/desc/"
    # file = "[ocean_988]_[GPT3|text-davinci-002]_[non-index]_[mpi-style-revised]_[order-II].pt"
    # FOR CONSTRAINT
    folder = "Constraint/order-symmetry/gpt2-xl/non-index/desc/"
    file = "[ocean_988]_[GPT2|gpt2-xl]_[non-index]_[mpi-style]_[order-I]_[desc].pt"
    mpi = load_mpi_instance(mpi_dir + folder + file)
    ocean_row = format_ocean_latex_table(mpi)
    print(ocean_row)
    ans_dist_row = format_ans_distribution_latex_table(mpi)
    print(ans_dist_row)
