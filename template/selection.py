
import os
import sys
import yaml
import colored
import argparse
from itertools import filterfalse
from matplotlib import pyplot as plt
from colorama import Fore, Back, Style

# Required for project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MPI.mpi import *
from util.utils import *
from template.templates import *
from model.language_model import *
from template.scoring import mutual_information


def main():
    # DEVICE Configuration
    print(colored.fg("#ffbf00") + Style.BRIGHT + line(n=120, is_print=False))
    if torch.backends.mps.is_available():
        print("-- MPS is built: ", torch.backends.mps.is_built())
        # See whether the following line is bug-free
        # print(
        #     f"-- Device Total Memory: {torch.mps.driver_allocated_memory() / (1024**2)}")
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
    parser.add_argument('--temp_idx', help='template index', type=int)
    parser.add_argument('--template', help='template string', type=str)
    parser.add_argument('--seed', help='python seed', type=int, default=2023)
    parser.add_argument('--verbose', help='verbose mode', action='store_true')
    parser.add_argument('--tag', help='tags', type=str, default='')
    parser.add_argument('--version', help='model version', type=str)
    args = parser.parse_args()

    # TO DELETE
    if not args.config:
        args.verbose = True
        args.config = 'config/template-selection/index/GPT2.yaml'
        args.version = 'gpt2-xl'
        args.template = '[og]-[s]-[type-iii]-[ans-iii]'

    assert args.config is not None, 'Please specify the config .yaml file to proceed.'
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if args.version:
        config['model']['version'] = args.version

    # PARSE YAML FILE
    # Set Seed if necessary
    set_seed(args.seed)
    #####  Experiment Type  #####
    regime, category = config['experiment'].values()
    assert regime in ['Constraint', 'Open-Vocab']
    assert category in ['order-symmetry',
                        'template-selection', 'prompt-engineering']

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
        # model = torch.nn.DataParallel(model)  # Default argument is fine
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
    if family in ['OPT', 'GPTNEO', 'GPTNEOX', 'BART', 'FLAN-T5', 'T0']:
        version = version.split('/')[1]

    log_dir += f"{regime}/{category}/{version}/{tmp['description']}/{ans_type}/"
    ckpt_dir += f"{regime}/{category}/{version}/{tmp['description']}/{ans_type}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    #####  logging filename  #####
    filename = log_fname(dset, model_config, tmp['description'])
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

    path = r"template/candidates/"
    templates = np.array(os.listdir(path))
    templates = list(filterfalse(lambda x: '.txt' not in x, templates))
    print(f"Here is a list of {len(templates)} candidate templates.")
    scores = {}

    is_lower_choices = [True, False]

    if args.temp_idx is not None:
        templates=[templates[args.temp_idx]]
        print('\nSingle Template Mode\n')
    elif args.template is not None:
        templates=[args.template]
        is_lower_choices = [True] if args.template[1:3]=='lc' else [False]
        print('\nSingle Template Mode\n')

    for tmp in tqdm(templates):
        for is_lower in is_lower_choices:
            ic(is_lower)
            fname = filename
            tmp_name = tmp.rstrip(".txt")
            # ic(tmp)
            prompt_template = get_template(tmp_name)
            if is_lower:
                tmp_name = tmp_name.replace('og', 'lc')
            fname += f"_{tmp_name}"
            # print(
            #     f">> TEMPLATE TESTING: {tmp_name} | LOWER CASE VERSION? {is_lower}")
            print(
                f">> TEMPLATE TESTING: {tmp_name}")
            ic(desc)
            # ic(prompt_template)
            mpi = MPI(path_to_dset, start, end,
                      prompt_template, index, desc, ans_type, is_lower,
                      regime, order, shuffle_both, verbose)
            mpi.reset()

            # Answer questions and calculate scores
            if regime == "Constraint":
                mpi.constraint_answer(
                    tokenizer, model, model_config, ll_type, half_precision, verbose)
                scores[tmp_name] = mutual_information(mpi.likelihood)
            elif regime == "Open-Vocab":
                assert generation_config is not None
                mpi.open_vocab_answer(tokenizer, model, model_config,
                                      generation_config, verbose)
                scores[tmp_name] = f"{(sum(mpi.valid_mask)* 100) / len(mpi.valid_mask)}%"
            else:
                assert False, 'Unrecognized Regime.'

            # Save metadata
            mpi.write_statistic(log_dir + fname + '.txt')
            mpi.save_checkpoint(ckpt_dir + fname + '.pt')

    if not (args.template or args.temp_idx):
        # For pretty print
        scores_df = pd.DataFrame.from_dict(
            scores, orient='index', columns=['Scores'])
        scores_df.reset_index(inplace=True)
        scores_df = scores_df.rename(columns={'index': 'Template'})
        scores_df = scores_df.sort_values(by=['Scores'], ascending=False)
        scores_df.reset_index(inplace=True)
        scores_df.to_csv(ckpt_dir + 'scores.csv')
        # Make plots
        plt.plot(np.arange(1, 37, 1),
                scores_df['Scores'], marker='s', color='navy', markersize=4)
        # plt.legend()
        plt.ylim(bottom=0.0)
        # plt.show()
        plt.xlabel("Rank")
        plt.ylabel("Scores")
        plt.title("Template Score vs. Rank")
        plt.savefig(log_dir + "score_vs_rank.png", dpi=400)
        plt.close()
        # Write files
        with open(log_dir + "scores.txt", 'w') as f:
            f.write("Template Selection Results\n")
            f.write(tabulate(scores_df[['Template', 'Scores']], headers='keys',
                    tablefmt='psql', showindex=True))
            f.write("\n")
            best_template = scores_df['Template'].loc[scores_df['Scores'].idxmax()]
            f.write(
                f"\nFor {family}, among these templates, {best_template} achieved the highest score.")


if __name__ == '__main__':
    main()
