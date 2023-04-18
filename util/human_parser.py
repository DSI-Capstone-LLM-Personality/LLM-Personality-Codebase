import pandas as pd
import numpy as np
import pyreadstat
from collections import defaultdict, Counter
from template.template import MPI_DESC
from tqdm import tqdm
import os
import time

# Human data parser


def process_raw_por(filename):
    start = time.time()
    df, meta = pyreadstat.read_por(f"Dataset/Human Data/{filename}.por")
    print(df.head())
    df.to_csv(f"Dataset/Human Data/{filename}.csv")
    stop = time.time()
    print(
        f"Processing Time: {np.round(stop - start, 2)}s | About {np.round((stop-start)/60, 1)} mins")

# process_raw_por("IPIP120")
# process_raw_por("IPIP300")


SCORES_TO_ANSWERS = {
    "+": {
        5: "Very Accurate",
        4: "Moderately Accurate",
        3: "Neither Accurate Nor Inaccurate",
        2: "Moderately Inaccurate",
        1: "Very Inaccurate"
    },
    "-": {
        1: "Very Accurate",
        2: "Moderately Accurate",
        3: "Neither Accurate Nor Inaccurate",
        4: "Moderately Inaccurate",
        5: "Very Inaccurate"
    }
}

######  Process Human Data  ######


def get_item_key_map(df, dset):
    assert dset in [120, 300]
    item_key_map = {}
    if dset == 120:
        tmp = df[['Short#', 'Sign']].loc[0:dset-1]
    else:
        tmp = df[['Full#', 'Sign']].loc[0:dset-1]
    # process
    for idx, trait in np.array(tmp):
        item_key_map["I" + str(idx)] = (trait[0], trait[1])
    # print(item_key_map)
    return item_key_map


def process_dset(df, ik_map):
    answer_distribution = {
        "+": {
            'O': defaultdict(int),
            'C': defaultdict(int),
            'E': defaultdict(int),
            'A': defaultdict(int),
            'N': defaultdict(int)
        },
        "-": {
            'O': defaultdict(int),
            'C': defaultdict(int),
            'E': defaultdict(int),
            'A': defaultdict(int),
            'N': defaultdict(int)
        }}
    coi = [f"I{i+1}" for i in range(len(ik_map))]
    df = df[coi]
    print(f"There are {len(df)} test takers.")
    print(f"There are {len(ik_map)} items.")
    print(f"Ideally, there are {len(df) * len(ik_map)} responses in total.")
    count, invalid_count = 0, 0
    for item in tqdm(coi):
        item_response = np.array(df[item])
        key, trait = ik_map[item]
        # print(key, trait, len(item_response))
        for response in item_response:
            if response not in range(1, 6, 1):
                invalid_count += 1
                continue
            answer_distribution[key][trait][SCORES_TO_ANSWERS[key]
                                            [int(response)]] += 1
            count += 1
        # break
    print(f"There are {invalid_count} responses that are problematic.")
    print(f"After discarding them, there are {count} responses in total.")
    print(answer_distribution['+']["C"])
    print(answer_distribution['-']["C"])
    return answer_distribution


def process_answer_distribution(answers, trait=None):
    assert trait is None or trait in ['O', 'C', 'E', 'A', 'N']
    if trait is None:
        ans_dist = {"+": defaultdict(int), "-": defaultdict(int)}
        for sign in ['+', '-']:
            for trait in ['O', 'C', 'E', 'A', 'N']:
                for item in MPI_DESC:
                    ans_dist[sign][item] += answers[sign][trait][item]
        return ans_dist
    else:
        return {"+": answers['+'][trait], "-": answers['-'][trait]}


def display_answer_distribution(answers):
    l = max(7, max([len(x) for x in MPI_DESC]))
    for sign in ['+', '-']:
        stat = Counter(answers[sign])
        print(f"{sign} Questions: ")
        print(f"{'ANSWERS':<{l}} | Count")
        for item in MPI_DESC:
            print(f"{item:<{l}} |   {stat[item]}")


qt_df = pd.read_excel('Dataset/Human Data/IPIP-NEO-ItemKey.xls')
df = pd.read_csv('Dataset/Human Data/IPIP300.csv')
item_key_map = get_item_key_map(qt_df, 300)
answer_distribution = process_dset(df, item_key_map)
# print(answer_distribution)
print("\nOVERALL STATISTICS")
overall = process_answer_distribution(answer_distribution)
display_answer_distribution(overall)
for trait in ['O', 'C', 'E', 'A', 'N']:
    print(f"\n\nTRAIT: {trait} | ANSWER DISTRIBUTION")
    trait_level_answer_distribution = process_answer_distribution(
        answer_distribution, trait)
    display_answer_distribution(trait_level_answer_distribution)
