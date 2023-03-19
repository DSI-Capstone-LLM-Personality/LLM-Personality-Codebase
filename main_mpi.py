from MPI.mpi import *
from Model.lm import *
from Model.template import *
import yaml
import random
import argparse

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
ans_type = tmp['ans_type']
# Shuffle
order_name = config['shuffle']['order']
if order_name is not None:
    order = ORDERS[order_name]
    shuffle_both = config['shuffle']['shuffle_both']
else:
    order, shuffle_both = None, None
#####  Model & Tokenizer  #####
model_config = config['model']
family, version = model_config['family'], model_config['version']
model = MODEL[family].from_pretrained(version, is_decoder=False)
tokenizer = TOKENIZER[family].from_pretrained(version)
#####  Likelihood calculation algorithm  #####
ll_type = config['algorithm']['ll_type']
#####  logging filename  #####
filename = log_fname(dset, model_config, tmp['description'])
if args.tag:
    tag = args.tag
    filename += f'_[{tag}]'
verbose = args.verbose
if order_name is not None:
    filename += f'_[{order_name}]'
    # ----------------------------------- #
    # ---------------- RUN -------------- #
mpi = MPI(path_to_dset, start, end,
          prompt, index, desc, ans_type, order, shuffle_both)
mpi.reset()
mpi.answer(tokenizer, model, model_config, ll_type, verbose)
mpi.write_statistic(log_dir + filename + '.txt')
mpi.save_checkpoint(ckpt_dir + filename + '.pt')
