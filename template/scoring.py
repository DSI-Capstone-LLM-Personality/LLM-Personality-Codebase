import os
import sys
import torch
import argparse
import numpy as np
from icecream import ic

# Required for project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MPI.mpi import *


def entropy(arr): return -np.sum(arr * np.log(arr), axis=1)

def normalize(arr): return arr / np.sum(arr, axis=1, keepdims=True)


def normalize1(arr):
    return np.exp(arr) / np.sum(np.exp(arr), axis=1, keepdims=True)


def mutual_information(x):
    # 50 x 5
    # input: 2D list of negative log of perplexity
    x = torch.stack(x, dim=0).numpy()
    x = 1 / np.exp(-x)
    # ic(x.shape)
    # this is only for MPI 5 options templates
    assert len(x.shape) == 2 and x.shape[1] == 5
    # ic(x)
    # ic(normalize1(x))
    x = normalize(x)
    # ic(x)
    h_y = entropy(np.mean(x, axis=0, keepdims=True)).item()
    h_y_given_x = np.mean(entropy(x))
    return h_y - h_y_given_x


class MIScorer():
    # This class is not necessary, remove later
    def __init__(self, ckpt_dir, ckpt_name):
        filename = ckpt_dir + ckpt_name
        self.ckpt = torch.load(filename)
        #

        # TODO: (Xiaoyang) this can be modified to be more generic...

    def __call__(self, x=None):
        if x is not None:
            print("Calculating Mutual Information of given probability vector...")
        else:
            x = torch.stack(self.ckpt.likelihood, dim=0).numpy()
            ic(x.shape)
            # this is only for MPI 5 options templates
            assert len(x.shape) == 2 and x.shape[1] == 5
            x = normalize(x)
        h_y = entropy(np.mean(x, axis=0, keepdims=True)).item()
        h_y_given_x = np.mean(entropy(x))
        return h_y - h_y_given_x


if __name__ == "__main__":

    # Simple test
    # ckpt_dir = "checkpoint/mpis/Constraint/order-symmetry/bert-base-uncased/non-index/desc/"
    # # ckpt_name = "[ocean_988]_[BERT|bert-base-uncased]_[non-index]_[order-III].pt"
    # ckpt_name = "[ocean_988]_[BERT|bert-base-uncased]_[non-index]_[original].pt"

    # mi_scorer = MIScorer(ckpt_dir, ckpt_name)
    # mi = mi_scorer()
    # print(mi)

    # Parse Input Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--chpk-dir', help='path to checkpoints')
    args = parser.parse_args()

    # Test
    if not args.chpk_dir:
        args.chpk_dir = 'checkpoint/mpis/Constraint/template-selection/opt-66b/non-index/desc'

    checkpoints = os.listdir(args.chpk_dir)

    scores = dict()
    for chkp in checkpoints:
        mpi = torch.load(os.path.join(args.chpk_dir, chkp))
        template_name = chkp.split('_')[-1][:-3]
        scores[template_name] = mutual_information(mpi.likelihood)

    logs_dir = args.chpk_dir.replace('mpis','log')

    # For pretty print
    scores_df = pd.DataFrame.from_dict(
        scores, orient='index', columns=['Scores'])
    scores_df.reset_index(inplace=True)
    scores_df = scores_df.rename(columns={'index': 'Template'})
    scores_df = scores_df.sort_values(by=['Scores'], ascending=False)
    scores_df.reset_index(inplace=True)
    scores_df.to_csv(os.path.join(args.chpk_dir,'scores.csv'))
    # Make plots
    plt.plot(np.arange(1, 37, 1),
            scores_df['Scores'], marker='s', color='navy', markersize=4)
    # plt.legend()
    plt.ylim(bottom=0.0)
    # plt.show()
    plt.xlabel("Rank")
    plt.ylabel("Scores")
    plt.title("Template Score vs. Rank")
    plt.savefig(os.path.join(logs_dir,"score_vs_rank.png"), dpi=400)
    plt.close()
    # Write files
    with open(os.path.join(logs_dir, "scores.txt"), 'w') as f:
        f.write("Template Selection Results\n")
        f.write(tabulate(scores_df[['Template', 'Scores']], headers='keys',
                tablefmt='psql', showindex=True))
        f.write("\n")
        best_template = scores_df['Template'].loc[scores_df['Scores'].idxmax()]
        f.write(
            f"\nFor {mpi.model_desc['family']}, among these templates, {best_template} achieved the highest score.")
