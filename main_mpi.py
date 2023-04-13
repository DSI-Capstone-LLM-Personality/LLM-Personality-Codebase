import os
import argparse
import yaml
from MPI.mpi import *
from Model.language_model import *
from Model.template import *
from util.utils import *
import colored
import colorama
from colorama import Fore, Back, Style
# colorama.init(autoreset=True)
# DEVICE Configuration
print(colored.fg("#ffbf00") + Style.BRIGHT + line(n=120, is_print=False))
if torch.backends.mps.is_available():
    print("-- MPS is built: ", torch.backends.mps.is_built())
    print("-- Let's use GPUs!")
elif torch.cuda.is_available():
    print(f"-- Current Device: {torch.cuda.get_device_name(0)}")
    print(
        f"-- Device Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("-- Let's use", torch.cuda.device_count(), "GPUs!")
else:
    print("-- Unfortunately, we are only using CPUs now.")
line(n=120)
print(colored.fg("#d33682") + Style.NORMAL + line(n=120, is_print=False))
# Parse Input Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='configuration file')
parser.add_argument('--seed', help='python seed', type=int, default=2023)
parser.add_argument('--verbose', help='verbose mode', action='store_true')
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
# ic(order_name)

#####  Model & Tokenizer Initialization  #####
model_config = config['model']
family, version, access_method = model_config.values()
if access_method == "api":
    model, tokenizer = None, None
elif access_method == "hf":
    # TODO: Revise this structure
    # if family == "GPT2" and regime == "Open-Vocab":
    #     model = MODEL[regime][family].from_pretrained(
    #         version, is_decoder=False).to(DEVICE)
    model = MODEL[regime][family].from_pretrained(
        version, is_decoder=False).to(DEVICE)
    # model = torch.nn.DataParallel(model)  # Default argument is fine
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

#####  Additional directory parsing (For necessary model family only)  #####
if family in ['GPTNEO', 'GPTNEOX', 'BART', 'FLAN-T5']:
    version = version.split('/')[1]

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
