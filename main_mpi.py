from Prompt.mpi import *
import yaml
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    help='configuration file')
parser.add_argument('--gpu',
                    help='gpu device number',
                    type=str, default='0')
parser.add_argument('--efficient',
                    help='if True, enables gradient checkpointing',
                    action='store_true')
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
ic(config['dataset'])
