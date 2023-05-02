import torch
import numpy as np
from collections import Counter, defaultdict
from util.utils import *
from template.templates import *
import string
import yaml
import argparse

# CMD LINE Example:
# # python3 util/table_generator.py --config=config/Open-Vocab/order-symmetry/GPT3/indexed.yaml

TABLE_ORDER_NAME = {
    'original': 'Original Order',
    'reverse': 'Reverse Order',
    'order-I': 'Random Order I',
    'order-II': 'Random Order II',
    'order-III': 'Random Order III'
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
family, version, _ = model_config.values()

prompt_template = config['template']['prompt']
ans_type = config['template']['ans_type']
description = config['template']['description']

mpis_dir = f"checkpoint/mpis/{regime}/{category}/{version}/{description}/{ans_type}/"
filename = log_fname(dset, model_config, description) + \
    f"_[{prompt_template}]"


def generate_table(table: str):
    if table == "symmetry":
        out = f"\multirow{{5:}}{{*}}[-0.3em]{{\\makecell[c]{{{family} \\\\ (\\textsc{{{prompt_template}}})}}}} "
        for order in ORDERS.keys():
            ckpt = torch.load(f"{mpis_dir}{filename}_[{order}].pt")
            out += f"& \\textsc{{{TABLE_ORDER_NAME[order]}}}" + \
                format_ocean_latex_table(ckpt) + "\n"
    elif table == "ans_dist":
        out = f"\multirow{{5:}}{{*}}[-0.3em]{{\\makecell[c]{{{family} \\\\ (\\textsc{{{prompt_template}}})}}}} "
        for order in ORDERS.keys():
            ckpt = torch.load(f"{mpis_dir}{filename}_[{order}].pt")
            out += f"& \\textsc{{{TABLE_ORDER_NAME[order]}}}" + \
                format_ans_distribution_latex_table(ckpt) + "\n"
    else:
        assert False, 'Unrecognized Table Type.'
        # print(out)
    print(out)


# CURRENT CODE ONLY WORKS FOR SYMMETRY EXPERIMENT TABLE GENERATION
generate_table("symmetry")
# generate_table("ans_dist")
