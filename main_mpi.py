from MPI.mpi import *
from Model.language_model import *
from Model.template import *
from util.utils import *
import yaml
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Parse Input Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='configuration file')
parser.add_argument('--seed', help='python seed', type=int, default=2023)
parser.add_argument('--verbose', help='verbose mode', type=bool, default=False)
parser.add_argument('--tag', help='tags', type=str, default='')
args = parser.parse_args()

assert args.config is not None, 'Please specify the config .yml file to proceed.'
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
# PARSE YAML FILE
# Set Seed if necessary
set_seed(args.seed)
#####  Experiment Type  #####
regime, category = config['experiment'].values()
assert regime in ['Constraint', 'Open-Vocab']
assert category in ['order-symmetry', 'prompt-engineering']

#####  File Path  #####
path = config['path']
dset_dir = path['dset_dir']
ckpt_dir = path['mpi_ckpt_dir']
log_dir = path['mpi_log_dir']

#####  Dataset  #####
dataset = config['dataset']
dset, start, end = dataset['dset'], dataset['start'], dataset['end']
path_to_dset = dset_dir + f"{dset}.csv"

#####  Prompt & Answer Template  #####
tmp = config['template']
prompt = PROMPT_TEMPLATE[tmp['prompt']]
# Prepare Index
index = tmp['index']
if index is not None:
    index = {k: INDEX[v] for k, v in index.items()}
# Prepare Desc
assert tmp['desc'] is not None
desc = {k: DESC[v] for k, v in tmp['desc'].items()}
# Prepare Answer
# Note that for "Constraint Regime" this is used for QA
# But for "Open-Vocab Regime", this only dictates how answers are presented
ans_type = tmp['ans_type']
# Shuffle
order_name = config['shuffle']['order']
if order_name is not None:
    order = ORDERS[order_name]
    shuffle_both = config['shuffle']['shuffle_both']
else:
    order, shuffle_both = None, None


#####  Model & Tokenizer Initialization  #####
model_config = config['model']
family, version, access_method = model_config.values()
if access_method == "api":
    model, tokenizer = None, None
elif access_method == "hf":
    model = MODEL[family].from_pretrained(version, is_decoder=False)
    tokenizer = TOKENIZER[family].from_pretrained(version)
else:
    assert 'Unrecognized Access Method.'


#####  Process Regime-Specific Arguments #####
if regime == "Constraint":
    #####  Likelihood calculation algorithm  #####
    ll_type = config['algorithm']['ll_type']
elif regime == "Open-Vocab":
    generation_config = config['params']
else:
    assert False, 'Unrecognized Regime.'

#####  Process directory  #####
log_dir += f"{regime}/{category}/{version}/{tmp['description']}/{ans_type}/"
ckpt_dir += f"{regime}/{category}/{version}/{tmp['description']}/{ans_type}/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)


#####  logging filename  #####
filename = log_fname(dset, model_config, tmp['description'])
filename += f"_[{tmp['prompt']}]"
if args.tag:
    tag = args.tag
    filename += f'_[{tag}]'
verbose = args.verbose
if order_name is not None:
    filename += f'_[{order_name}]'
if ans_type is not None and regime == "Constraint":
    filename += f"_[{ans_type}]"

# ----------------------------------- #
# ---------------- RUN -------------- #
mpi = MPI(path_to_dset, start, end,
          prompt, index, desc, ans_type,
          regime, order, shuffle_both)
mpi.reset()

if regime == "Constraint":
    mpi.constraint_answer(tokenizer, model, model_config, ll_type, verbose)
elif regime == "Open-Vocab":
    assert generation_config is not None
    mpi.open_vocab_answer(tokenizer, model, model_config,
                          generation_config, verbose)
else:
    assert False, 'Unrecognized Regime.'

mpi.write_statistic(log_dir + filename + '.txt')
mpi.save_checkpoint(ckpt_dir + filename + '.pt')
