import os
import yaml
import colored
import argparse
from MPI.mpi import *
from util.utils import *
from template.templates import *
from model.language_model import *
from colorama import Fore, Back, Style


def main_mpi(args, config):
   
    # DEVICE Configuration
    print(colored.fg("#ffbf00") + Style.BRIGHT + line(n=120, is_print=False))

    if torch.cuda.is_available():
        print(f"-- Current Device: {torch.cuda.get_device_name(0)}")
        print(f"-- Device Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print("-- Let's use", torch.cuda.device_count(), "GPUs!")
    line(n=120)
    print(colored.fg("#d33682") + Style.NORMAL + line(n=120, is_print=False))   

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
    prompt_template = get_template(tmp['prompt'])
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
    # All .yaml are deprecated rather than the one for GPT constraint search
    is_lower = tmp['is_lower_case']
    # Shuffle
    order_name = config['shuffle']['order']
    if order_name is not None:
        order = ORDERS[order_name]
        shuffle_both = config['shuffle']['shuffle_both']
    else:
        order, shuffle_both = None, None
    # ic(order_name)

    #####  model & Tokenizer Initialization  #####
    model_config = config['model']
    family, version, access_method, half_precision = model_config.values()
    if access_method == "api":
        model, tokenizer = None, None
    elif access_method == "hf":
        # TODO: Revise this structure
        # if family == "GPT2" and regime == "Open-Vocab":
        #     model = MODEL[regime][family].from_pretrained(
        #         version, is_decoder=False).to(DEVICE)
        tokenizer = TOKENIZER[family].from_pretrained(version)
        if half_precision and DEVICE.lower() != 'cpu':
            model = MODEL[regime][family].from_pretrained(
                version, pad_token_id=tokenizer.eos_token_id).half().to(DEVICE)
        else:
            model = MODEL[regime][family].from_pretrained(
                version, pad_token_id=tokenizer.eos_token_id).to(DEVICE)
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
    if family in ['GPTNEO', 'GPTNEOX', 'BART', 'FLAN-T5', 'T0', 'OPT']:
        version = version.split('/')[1]

    log_dir += f"{regime}/{category}/{version}/{tmp['description']}/{ans_type}/"
    ckpt_dir += f"{regime}/{category}/{version}/{tmp['description']}/{ans_type}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)


    #####  logging filename  #####
    filename = log_fname(dset, model_config, tmp['description'])
    tmp_name = tmp['prompt']
    if is_lower:
        tmp_name = tmp_name.replace('og', 'lc')
    filename += f"_{tmp_name}"

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
            prompt_template, index, desc, ans_type, is_lower,
            regime, order, shuffle_both, verbose)
    mpi.reset()

    if regime == "Constraint":
        mpi.constraint_answer(tokenizer, model, model_config, ll_type, half_precision, verbose)
    elif regime == "Open-Vocab":
        assert generation_config is not None
        mpi.open_vocab_answer(tokenizer, model, model_config,
                            generation_config, verbose)
    else:
        assert False, 'Unrecognized Regime.'


    mpi.write_statistic(log_dir + filename + '.txt')
    mpi.save_checkpoint(ckpt_dir + filename + '.pt')


if __name__=='__main__':

    # Parse Input Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file')
    parser.add_argument('--order', help='order')
    parser.add_argument('--ans', help='ans_type')
    parser.add_argument('--seed', help='python seed', type=int, default=2023)
    parser.add_argument('--verbose', help='verbose mode', action='store_true')
    parser.add_argument('--tag', help='tags', type=str, default='')
    args = parser.parse_args()

    # To Delete
    # args.config = 'config/Constraint/order-symmetry/GPT2-Medium/non-index.yaml'

    config = yaml.load(open(os.path.join('config/Constraint/order-symmetry',args.config,'index.yaml'), 'r'), Loader=yaml.FullLoader)

    if args.ans:
        config['template']['ans_type'] = args.ans
        # config['template']['description'] = args.ans
        if args.order:
            config['shuffle']['order']= args.order
            main_mpi(args,config)
        else:
            orders = ["original", "reverse", "order-I", "order-II", "order-III"]
            for order in orders:
                config['shuffle']['order']= order
                main_mpi(args,config)
    else:
        ans_types = ['index', 'index-desc', 'desc']
        for ans in ans_types:
            config['template']['ans_type'] = ans
            # config['template']['description'] = ans
            if args.order:
                config['shuffle']['order'] = args.order
                main_mpi(args,config)
            else:
                orders = ["original", "reverse", "order-I", "order-II", "order-III"]
                for order in orders:
                    config['shuffle']['order'] = order
                    main_mpi(args,config)