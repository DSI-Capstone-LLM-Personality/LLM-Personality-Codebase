from MPI.mpi import *
from Model.lm import *
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
if args.seed:
    seed = args.seed
    # print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
option = {k: ANSWER_TEMPLATE[v] for k, v in tmp['option'].items()}
answer = {k: ANSWER_TEMPLATE[v] for k, v in tmp['answer'].items()}
# Shuffle
shuffle = config['shuffle']['status']
identifier = config['shuffle']['identifier']
#####  Model & Tokenizer  #####
model_config = config['model']
family, version = model_config['family'], model_config['version']
model = MODEL[family].from_pretrained(version)
tokenizer = TOKENIZER[family].from_pretrained(version)
#####  Likelihood calculation algorithm  #####
ll_type = config['algorithm']['ll_type']
#####  logging filename  #####
filename = log_fname(dset, model_config, tmp['description'])
if shuffle:
    assert identifier is not None, 'Please specify a file identifier'
    filename += f'_[{identifier}]'
if args.tag:
    tag = args.tag
    filename += f'_[{tag}]'
verbose = args.verbose

# ----------------------------------- #
# ---------------- RUN -------------- #
mpi = MPI(path_to_dset, start, end,
          prompt, option, answer, shuffle)
mpi.reset()
mpi.answer(tokenizer, model, model_config, ll_type, verbose)
mpi.write_statistic(log_dir + filename + '.txt')
mpi.save_checkpoint(ckpt_dir + filename + '.pt')
