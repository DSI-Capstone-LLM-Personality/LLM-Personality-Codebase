# ------------------------- #
# SCRIPT FOR MPI EXPERIMENT #
# AUTHOR: XIAOYANG SONG     #
# ------------------------- #
from functools import reduce
from matplotlib import pyplot as plt
from regex import E
import re
import pandas as pd
from collections import Counter, defaultdict
from util.utils import *
from tqdm import tqdm
import sys
from colorama import Fore, Back, Style
import colored
from tabulate import tabulate
from template.templates import *
from model.language_model import *

# Some Abbreviations:
# ans -> "answer", inv -> "inverse", perp -> "perplexity"


def read_mpi(path, show=False, n=None, verbose=False):
    df = pd.read_csv(path)
    if show:
        assert n is not None, 'Please specify how many rows you want to view.'
        ic(df.head(n))
    if verbose:
        print(f"Personality Test on MPI dataset...")
        print(f"There are {len(df)} multiple choice questions in total.")
    return df


def check_column_cleanness(df, col_name):
    col = df[col_name]
    ic(Counter(col))
    ic(len(Counter(col)))
    if col_name == 'label_ocean':
        assert len(Counter(col)) == 5
    elif col_name == 'key':
        assert len(Counter(col)) == 2
    # TODO: (Xiaoyang) add more cleanness check later...


class MPI():
    # TODO: (Xiaoyang) [TOP PRIORITY] Re-structure this class
    # (this should also work for non-mpi templates)
    def __init__(self, path_to_file, start, end,
                 prompt, index, desc, ans_type, is_lower_case,
                 regime="constraint", order=None, shuffle_both=None, verbose=False):
        self.mpi_df = read_mpi(path_to_file)

        # (Optional): only testing the first few examples
        if start is not None and end is not None:
            self.mpi_df = self.mpi_df[start: end]

        # LABEL, KEY & + -
        self.label = np.array(self.mpi_df['label_ocean'])
        self.key = torch.tensor(list(self.mpi_df['key']), dtype=torch.long)
        self.plus_minus = np.array(["+" if k == 1 else "-" for k in self.key])

        # STATEMENT
        self.text = np.array(self.mpi_df['text'])

        # PROMPT
        self.prompt, self.order, self.shuffle_both = prompt, order, shuffle_both
        self.index, self.desc = index, desc

        # OPTIONS
        self.is_lower_case = is_lower_case
        self.option_formatter = MPIOptionFormatter(
            self.index, self.desc, self.is_lower_case)
        self.option = self.option_formatter(order, shuffle_both)

        # Print out options for checking purpose
        if verbose:
            line(120)
            print(colored.fg("#00b384") + Style.BRIGHT +
                  line(n=120, is_print=False))
            for key, vals in self.option.items():
                print(colored.fg('#ffbf00') + f"OPTIONS for {key} QUESTIONS: ")
                print(colored.fg("#00b384") + f">> {vals}")
        # ANSWERS
        self.mpi_choice_lst = MPI_options_to_answers(
            self.index, self.desc, self.option, ans_type, is_lower_case, order)

        if verbose:
            line(120)
            for key, vals in self.mpi_choice_lst.items():
                print(colored.fg('#ffbf00') + f"ANSWERS for {key} QUESTIONS: ")
                print(colored.fg("#00b384") + f">> {vals}")
            line(120)

        # QUESTIONS
        self.formatter = MPIQuestionFormatter(prompt, self.option)
        self.questions = np.array([self.formatter(x, k)
                                   for x, k in zip(self.text, self.plus_minus)])
        # OCEAN SCORE
        self.OCEAN, self.scores = defaultdict(list), []

        # EXPERIMENT TYPE
        self.regime = regime

        # META-DATA: probability, likelihood, etc...
        if self.regime == "Constraint":
            self.likelihood, self.probs, self.token_of_interest = [], [], []
        else:
            self.prompter, self.processor = None, PROCESSER(
                self.mpi_choice_lst['+'])
            self.raw_response, self.processed_response, self.mpi_response = [], [], []
            self.valid_mask = None

        # Results
        self.preds_key, self.preds = [], []
        self.answered, self.model_desc = False, None

        # SANITY CHECK CODE (Optional)
        # check_column_cleanness(self.mpi_df, 'label_ocean')
        # check_column_cleanness(self.mpi_df, 'key')

    def reset(self):
        self.OCEAN = defaultdict(list)
        self.scores = []
        if self.regime == "Constraint":
            self.likelihood, self.probs, self.token_of_interest = [], [], []
        else:
            self.prompter, self.processor = None, PROCESSER(
                self.mpi_choice_lst['+'])
            self.raw_response, self.processed_response, self.mpi_response = [], [], []
            self.valid_mask = None
        self.preds_key, self.preds = [], []
        self.answered = False
        self.model_desc = None
        # TODO: (Xiaoyang) more functionality here...

    def open_vocab_answer(self, tokenizer, model, model_desc: dict, param_dict: dict, verbose=False):
        print(colored.fg("blue")+line(120, False))
        print(colored.fg("blue")+"Open Vocabulary Search Experiment Running......")
        print(colored.fg("blue")+line(120, False))

        if verbose:
            self.display_sample_questions(self.questions[0], self.regime)
        assert not self.answered
        assert "version" in model_desc
        assert "family" in model_desc

        # TODO: add cases
        family, version = model_desc['family'], model_desc['version']
        self.prompter = PROMPTER(family, model, tokenizer, param_dict, version)

        # TODO: use the parser class here
        with torch.no_grad():
            for idx, prompt in enumerate(tqdm(self.questions, colour="#b58900")):
                key = self.plus_minus[idx]
                response = self.prompter(prompt)
                # Process generated responses
                processed_response, pred = self.processor(response)
                # print(response)
                # TODO: (Xiaoyang) add error handling: MPI scoring mechanism might be incorrect
                try:
                    mpi_response = re.search(
                        r'[abcdeABCDE][^a-zA-Z]', response + ')', flags=0).group()[0].upper()
                except Exception:
                    print("MPI scoring mechanism fails...")
                    mpi_response = "UNK"

                # THIS PART IS ONLY USEFUL WHEN REPRODUCING MPI PAPER RESULTS
                MPI = False
                if MPI:
                    if mpi_response != 'UNK' and mpi_response in LETTER:
                        pred = list(LETTER).index(mpi_response)
                    else:
                        pred = -1
                # STORE STATISTICS
                self.preds.append(pred)
                self.raw_response.append(response)
                self.processed_response.append(processed_response)
                self.mpi_response.append(mpi_response)
                if pred == -1:
                    if verbose:
                        print(colored.fg('#d33682') + line(120, False))
                        print(
                            f"QUESTION #{idx:<4} | TRAIT: {self.label[idx]} | KEY: {key}")
                        print(f">> Generated Response:\n{response}")
                        print(f"-- MPI ANSWER: {mpi_response}")
                        print(
                            f"THIS QUESTION IS DISCARDED! GENERATED RESPONSE IS NOT VALID.")
                    self.preds_key.append("Not Valid")
                    self.scores.append(-1)
                    continue
                else:
                    self.preds_key.append(self.mpi_choice_lst[key][pred])
                    score = MPI_SCORE[key][pred]
                    self.scores.append(score)
                if verbose:
                    print(
                        f"\nQUESTION #{idx:<4} | TRAIT: {self.label[idx]} | KEY: {key} | SCORE: {score}")
                    print(f">> Generated Response: {response}")
                    print(
                        f"-- Processed Response (OURS): {processed_response}")
                    print(f"-- OUR ANSWER: {self.mpi_choice_lst[key][pred]}")
                    print(f"-- MPI ANSWER: {mpi_response}")

            # SCORE CALCULATION
            # Valid mask: filter out invalid response
            self.valid_mask = self.processor.valid_idx
            self.preds_key = np.array(self.preds_key)
            self.answered = True
            self.model_desc = model_desc
            self.scores = np.array(self.scores)
            self.calculate_score()
            if verbose:
                self.display_ocean_stats()
                self.display_aux_stats()
                self.display_trait_stats()

    def constraint_answer(self, tokenizer, model, model_desc: dict, ll_type="ans_inv_perp", half_precision=False, verbose=False):

        print(colored.fg("blue")+line(120, False))
        print(colored.fg("blue")+"Constraint Search Experiment Running......")
        print(colored.fg("blue")+line(120, False))

        # Argument check
        assert not self.answered
        assert "version" in model_desc
        assert "family" in model_desc
        family = model_desc['family']  # TODO: add cases
        prober = LMPROB(family, model, tokenizer, ll_type, half_precision)

        if verbose:
            self.display_sample_questions(self.questions[0], self.regime)
            # print(colored.fg('#b58900') + line(120, False))
        with torch.no_grad():
            for idx, prompt in enumerate(tqdm(self.questions, colour='#b58900')):
                ll_lst, prob_lst, toi_lst = [], [], []
                # NOTE: currently unequal sequence length forbids us doing batch-level operation
                # TODO: (Xiaoyang) Find a way to do batch-level processing later...
                key = self.plus_minus[idx]
                for choice in self.mpi_choice_lst[key]:
                    prob, ll, toi = prober(prompt, choice)
                    ll_lst.append(ll.item())
                    # PROBABILITY FOR EACH WORD IN THE SENTENCE
                    prob_lst.append(prob)
                    # TOKEN OF INTERESTS
                    toi_lst.append(tokenizer.decode(toi).strip())
                # MCQA BASED ON LIKELIHOOD
                ll_lst = torch.tensor(ll_lst)
                pred = torch.argmax(ll_lst).item()
                # SAVE STATISTICS
                self.token_of_interest.append(toi_lst)
                self.likelihood.append(ll_lst)
                self.probs.append(prob_lst)
                # SAVE MCQA-ANSWERS
                self.preds_key.append(self.mpi_choice_lst[key][pred])
                self.preds.append(pred)
                # TODO: (Xiaoyang) THERE IS A BUG when shuffling orderes. Fix this bug later...
                score = MPI_SCORE[key][pred]
                self.scores.append(score)

                if verbose:
                    print(colored.fg('#d33682') + line(120, False))
                    print(
                        f"QUESTION #{idx:<4} | TRAIT: {self.label[idx]} | KEY: {key} | SCORE: {score} | ANSWER: {self.mpi_choice_lst[key][pred]}")
                    print(
                        f"-- Inverse Log-Perplexity: {list(np.round(np.array(ll_lst), 4))}")
                    print(f"-- Tokens of Interests: {toi_lst}")

            # SCORE CALCULATION
            self.preds_key = np.array(self.preds_key)
            self.calculate_score()
            # SET MPI status
            self.answered = True
            self.model_desc = model_desc

            if verbose:
                self.display_ocean_stats()
                self.display_aux_stats()
                self.display_trait_stats()

    def calculate_score(self):
        if self.regime == "Open-Vocab":
            assert self.valid_mask is not None
            scores = self.scores[self.valid_mask]
            labels = self.label[self.valid_mask]
            assert len(scores) == len(labels)
            assert -1 not in scores
        else:
            labels, scores = self.label, self.scores
        # Calculating scores
        for idx, score in enumerate(scores):
            self.OCEAN[labels[idx]].append(score)

    @staticmethod
    def display_sample_questions(question, regime):
        line()
        print(f"Sample questions look like this:\n{question}")
        line()
        print(f"{regime} MCQA task starts...")
        line()

    def display_ocean_stats(self):
        # ic(self.OCEAN)
        line()
        if self.regime == "Open-Vocab":
            print(
                f"There are {len(self.scores[self.valid_mask])} questions with valid response in total.")
        print("OCEAN SCORES STATS")
        self.stats = {}
        for item in OCEAN:
            vals = torch.tensor(self.OCEAN[item], dtype=torch.float32)
            mean, std = torch.mean(vals).item(), torch.std(vals).item()
            self.stats[item] = [mean, std]
            print(
                f"{item} | MEAN: {np.round(mean, 5):<8} | STD: {np.round(std, 5)}")
        # TODO: (Xiaoyang) add more functionality here
        # self.reset()
        # return np.array(self.stats)
        # return self.stats

    def display_aux_stats(self):
        line()
        print("OTHER INTERESTING STATS")
        # Format length
        l = max(7, max([len(x) for x in self.mpi_choice_lst['+']]))
        for sign in ['+', '-']:
            condition_mask = self.plus_minus == sign
            if self.regime == "Constraint":
                mask = condition_mask
            else:
                mask = np.logical_and(condition_mask, self.valid_mask)
            stat = Counter(self.preds_key[mask])
            print(f"{sign} Questions: ")
            print(f"{'ANSWERS':<{l}} | Count")
            for item in self.mpi_choice_lst[sign]:
                print(f"{item:<{l}} |  {stat[item]}")
        if self.regime != 'Constraint':
            self.processor.display_stats()

    def display_trait_stats(self):
        line()
        print("TRAITS-LEVEL STATS: ")
        score = list(np.arange(1, 6, 1))
        self.traits = {}
        for item in OCEAN:
            count = dict(Counter(np.array(self.OCEAN[item])))
            df = pd.DataFrame()
            df[score] = None
            # TODO:(Xiaoyang) revision later..
            count_arr = [count[x] if x in count else 0 for x in score]
            df.loc[len(df.index)] = count_arr
            self.traits[item] = count_arr
            print(f"Trait: {item} | # Questions: {np.sum(count_arr)}")
            # CHOICE DISTRIBUTION: + and -
            l = max(7, max([len(x) for x in self.mpi_choice_lst['+']]))

            # Depending on regimes: masked invalid questions

            for sign in ["+", "-"]:
                print(f"> CHOICES DISTRIBUTION [{sign}]")
                condition_mask = np.logical_and(self.label == item,
                                                self.plus_minus == sign)
                if self.regime == "Open-Vocab":
                    mask = np.logical_and(condition_mask, self.valid_mask)
                else:
                    mask = condition_mask
                stat = Counter(self.preds_key[mask])
                print(f"{'ANSWERS':<{l}} | Count")
                for choice in self.mpi_choice_lst[sign]:
                    print(f"{choice:<{l}} |   {stat[choice]}")
                print("")
                # Save distribution plot
                # plt.bar(['VA', 'MA', "NANI", "MI", "VI"], [stat[x] for x in self.mpi_choice_lst[sign]])
                # plt.xlabel("Options")
                # plt.ylabel("Count")
                # version = self.model_desc['version'].split('/')[1] 
                # plt.title(f"Choice Distribution - [{sign}] - Trait {item} ({version})")
                # plt.savefig(f"plot/distribution/{version}/{version}-[{sign}]-{item}.jpg", dpi=500)
                # plt.close()

            # SCORE DISTRIBUTION
            print("> SCORE DISTRIBUTION")
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
            print("\n")
        # return traits

    def write_statistic(self, filename):
        assert self.answered, 'Can not write statistics to files. Questions are not answered yet.'

        original_stdout = sys.stdout
        with open(filename, 'w') as f:
            sys.stdout = f
            # Write model information
            print(
                f'MODEL: {self.model_desc["family"]} | Version: {self.model_desc["version"]}')
            line()
            # Write sample question
            print(f"There are {len(self.mpi_df)} MC questions in total.")
            line()
            print(f"SHUFFLED? | {self.order}")
            if self.order is not None:
                print(
                    f"> Shuffle both indexes and descriptions?: {self.shuffle_both}")
            print(
                f"The question template look like this:\n\n{self.questions[0]}")
            line()
            if self.regime == "Constraint":
                for sign in ['+', '-']:
                    print(
                        f"The choice available for \"{sign}\" questions looks like this:\n> {list(self.mpi_choice_lst[sign])}")
                    line()
            print("ANSWER STATISTICS")
            self.display_ocean_stats()
            self.display_aux_stats()
            self.display_trait_stats()
            line()
            print("APPENDIX: ANSWERS")
            line()
            for idx, statement in enumerate(self.text):
                if self.regime == "Constraint":
                    print(
                        f"QUESTION #{idx+1:<4} | TRAIT: {self.label[idx]} | KEY: {self.plus_minus[idx]} | SCORE: {self.scores[idx]} | ANSWER: {self.preds_key[idx]}")
                    print(f"> Statement: {statement}")
                    print(
                        f"> Inverse Log-Perplexity: {list(np.round(np.array(self.likelihood[idx]), 4))}")
                else:
                    print(
                        f"QUESTION #{idx:<4} | TRAIT: {self.label[idx]} | KEY: {self.plus_minus[idx]} | SCORE: {self.scores[idx]}")
                    print(f"> Statement: {statement}")
                    print(f">> Generated Response: {self.raw_response[idx]}")
                    print(
                        f"-- Processed Response (OURS): {self.processed_response[idx]}")
                    print(
                        f"-- OUR ANSWER: {self.preds_key[idx]}")
                    print(f"-- MPI ANSWER: {self.mpi_response[idx]}")
                    if not self.valid_mask[idx]:
                        print(
                            f"THIS QUESTION IS DISCARDED! GENERATED RESPONSE IS NOT VALID!")
                    print("\n")
            line()
            # If necessary write statistics in latex format
            print(format_ocean_latex_table(self))
            print(format_ans_distribution_latex_table(self))
            f.close()
            sys.stdout = original_stdout

    def save_checkpoint(self, filename):
        torch.save(self, filename)


if __name__ == '__main__':
    ic("Prompting experiments on MPI dataset...")
    # Filename (may vary on your local computer)
    # filename = "mpi_small"
    # local_path = "Dataset/" + f"{filename}.csv"
    # read_mpi(local_path, True, 5)

    # mpi_dir = "checkpoint/mpis/"
    # folder = "Constraint/order-symmetry/gpt2-xl/non-index/desc/"
    # file = "[ocean_988]_[GPT2|gpt2-xl]_[non-index]_[mpi-style]_[order-I]_[desc].pt"
    # mpi = load_mpi_instance(mpi_dir + folder + file)
