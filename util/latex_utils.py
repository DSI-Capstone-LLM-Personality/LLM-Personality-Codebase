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
    'original': 'Original',
    'reverse': 'Reverse',
    'order-I': 'Order I',
    'order-II': 'Order II',
    'order-III': 'Order III'
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
# template_type = " \\textsc{{Non-Indexed}} "
template_type = " \\textsc{{Indexed}}"
# MANUALLY CHANGE THIS
# ans_type = 'index-desc'
ans_type='index'
# ans_type='desc'

calibrated = True
# calibrated=False

mpis_dir = f"checkpoint/mpis/{regime}/{category}/{version}/{description}/{ans_type}/"
filename = log_fname(dset, model_config, description)

tmp_name = prompt_template
if is_lower:
    tmp_name = tmp_name.replace('og', 'lc')
filename += f"_{tmp_name}"


#TODO: Clean later
def generate_table(table: str):
    if table == "symmetry":
        out = f"\multirow{{5}}{{*}}[-0.3em]{{\\makecell[c]{{{family} \\\\ \\textsc{{{tmp_name}}}}}}} "
        for order in ORDERS.keys():
            fname = f"{mpis_dir}{filename}_[{order}]"
            if ans_type is not None and regime == "Constraint":
                fname += f"_[{ans_type}]"
            if calibrated:
                fname += f"_[calibrated]"
            ckpt = torch.load(f"{fname}.pt",map_location=DEVICE)
            out += f"& \\textsc{{{TABLE_ORDER_NAME[order]}}}" + \
                format_ocean_latex_table(ckpt) + "\n"
        out += "\\midrule\n"
    elif table == "ans_dist":
        out = f"\multirow{{5}}{{*}}[-0.3em]{{\\makecell[c]{{{family} \\\\ \\textsc{{{tmp_name}}}}}}} "
        for order in ORDERS.keys():
            fname = f"{mpis_dir}{filename}_[{order}]"
            if ans_type is not None and regime == "Constraint":
                fname += f"_[{ans_type}]"
            if calibrated:
                fname += f"_[calibrated]"
            ckpt = torch.load(f"{fname}.pt",map_location=DEVICE)
            out += f"& \\textsc{{{TABLE_ORDER_NAME[order]}}}" + \
                format_ans_distribution_latex_table(ckpt) + "\n"
            # PLOT DISTRIBUTION
            # ckpt.display_trait_stats()
            # break
        out += "\\midrule\n"
    elif table == "score_dist":
        out = f"\multirow{{5}}{{*}}[-0.3em]{{{family}}} "
        # out += f"& \multirow{{5}}{{*}}[-0.3em]{{{template_type}}}"
        for order in ORDERS.keys():
            fname = f"{mpis_dir}{filename}_[{order}]"
            if ans_type is not None and regime == "Constraint":
                fname += f"_[{ans_type}]"
            if calibrated:
                fname += f"_[calibrated]"
            # For score distribution only
            # out += template_type
            ckpt = torch.load(f"{fname}.pt",map_location=DEVICE)
            # if order != 'original':
            #     out += " &"
            out += f" & \\textsc{{{TABLE_ORDER_NAME[order]}}}"
            out += format_score_distribution_latex_table(ckpt) + "\n"
            # PLOT DISTRIBUTION
            # ckpt.display_trait_stats()
            # break
        out += "\\midrule\n"
    else:
        assert False, 'Unrecognized Table Type.'
        # print(out)
    print(out)


# CURRENT CODE ONLY WORKS FOR SYMMETRY EXPERIMENT TABLE GENERATION
# generate_table("symmetry")
# generate_table("ans_dist")
generate_table("score_dist")

