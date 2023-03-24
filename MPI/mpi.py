# ------------------------- #
# SCRIPT FOR MPI EXPERIMENT #
# AUTHOR: XIAOYANG SONG     #
# ------------------------- #
from functools import reduce

from regex import E
from Prompt.bert_prompt import *
import pandas as pd
from collections import Counter, defaultdict
from utils import *
from tqdm import tqdm
import sys
from tabulate import tabulate
from Model.template import *
from Model.lm import *

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


def prepare_mpi_questions(statement):
    # TODO:(Xiaoyang) DEPRECATED FUNCTION: remove later...
    questions = f"Given a statement of you: \"You {statement}.\" "
    prompt = "Please choose from the following options to identify how accurately this statement describes you."
    options = "\nOptions: \n(A). Very Accurate \n(C). Neither Accurate Nor Inaccurate \n(D). Moderately Inaccurate \n(E). Very Inaccurate \n(B). Moderately Accurate \nAnswers: "
    # options = "\nOptions: \n(A). Very Accurate \n(B). Moderately Accurate \n(C). Neither Accurate Nor Inaccurate \n(D). Moderately Inaccurate \n(E). Very Inaccurate \nAnswers: "
    return questions + prompt + options
    # return questions


def check_column_cleanness(df, col_name):
    col = df[col_name]
    ic(Counter(col))
    ic(len(Counter(col)))
    if col_name == 'label_ocean':
        assert len(Counter(col)) == 5
    elif col_name == 'key':
        assert len(Counter(col)) == 2
    # TODO: (Xiaoyang) add more cleanness check later...


# TODO:(Xiaoyang) two functions that calculate likelihood...


def run_mpi(dset_config: dict,
            model_config: dict,
            algo_config: dict,
            template_config: dict,
            filename=None,
            verbose=False):
    # TODO: (Xiaoyang) deprecated function, remove later...
    # PARSE MPI Dataset information
    path_to_dset, start, end = dset_config.values()
    # PARSE targeting model information
    model, tokenizer, model_desc = model_config.values()
    # PARSE algorithm-level config
    ll_type = algo_config['ll_type']
    # PARSE template
    prompt, mpi_option, mpi_choice, shuffle = template_config.values()
    # RUN
    mpi = MPI(path_to_dset, start, end,
              prompt, mpi_option, mpi_choice, shuffle)
    mpi.reset()
    mpi.answer(tokenizer, model, model_desc, ll_type=ll_type, verbose=verbose)
    if filename is not None:
        mpi.write_statistic(filename)
    return mpi


class MPI():
    # TODO: (Xiaoyang) [TOP PRIORITY] Re-structure this class
    # (this should also work for non-mpi templates)
    def __init__(self, path_to_file, start, end,
                 prompt, index, desc, ans_type, order=None, shuffle_both=None):
        self.mpi_df = read_mpi(path_to_file)
        # (Optional): only testing the first few examples
        if start is not None and end is not None:
            self.mpi_df = self.mpi_df[start: end]
        # LABEL, KEY & + -
        self.label = np.array(self.mpi_df['label_ocean'])
        self.key = torch.tensor(self.mpi_df['key'], dtype=torch.long)
        self.plus_minus = np.array(["+" if k == 1 else "-" for k in self.key])
        # STATEMENT
        self.text = np.array(self.mpi_df['text'])
        # PROMPT
        self.prompt, self.order, self.shuffle_both = prompt, order, shuffle_both
        self.index, self.desc = index, desc
        # OPTIONS
        self.option_formatter = MPIOptionFormatter(self.index, self.desc)
        self.option = self.option_formatter(order, shuffle_both)
        ic(self.option)
        # ANSWERS
        self.mpi_choice_lst = MPI_options_to_answers(
            self.index, self.desc, self.option, ans_type, order)
        ic(self.mpi_choice_lst)
        # QUESTIONS
        self.formatter = MPIQuestionFormatter(prompt, self.option)
        self.questions = np.array([self.formatter(x, k)
                                  for x, k in zip(self.text, self.plus_minus)])
        # OCEAN SCORE
        self.OCEAN, self.scores = defaultdict(list), []
        # META-DATA: probability, likelihood, etc...
        self.likelihood, self.probs, self.token_of_interest = [], [], []
        self.preds_key, self.preds = [], []
        self.answered, self.model_desc = False, None

        # SANITY CHECK CODE (Optional)
        # check_column_cleanness(self.mpi_df, 'label_ocean')
        # check_column_cleanness(self.mpi_df, 'key')

    def reset(self):
        self.OCEAN = defaultdict(list)
        self.scores = []
        self.likelihood, self.probs, self.token_of_interest = [], [], []
        self.preds_key, self.preds = [], []
        self.answered = False
        self.model_desc = None
        # TODO: (Xiaoyang) more functionality here...

    def answer(self, tokenizer, model, model_desc: dict, ll_type="ans_inv_perp", verbose=False):
        # Argument check
        assert "version" in model_desc
        assert "family" in model_desc
        family = model_desc['family']  # TODO: add cases
        prober = PROBS(family, model, tokenizer, ll_type)
        if verbose:
            line()
            print(f"Sample questions look like this:\n{self.questions[0]}")
            line()
            print("MCQA task starts...")
            line()
        with torch.no_grad():
            for idx, prompt in enumerate(tqdm(self.questions)):
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
                    toi_lst.append(tokenizer.decode(toi))
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
        for idx, score in enumerate(self.scores):
            self.OCEAN[self.label[idx]].append(score)

    def display_ocean_stats(self):
        # ic(self.OCEAN)
        line()
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

    def display_aux_stats(self):
        line()
        print("OTHER INTERESTING STATS")
        # Format length
        l = max(7, max([len(x) for x in self.mpi_choice_lst['+']]))
        for sign in ['+', '-']:
            stat = Counter(self.preds_key[self.plus_minus == sign])
            print(f"{sign} Questions: ")
            print(f"{'ANSWERS':<{l}} | Count")
            for item in self.mpi_choice_lst[sign]:
                print(f"{item:<{l}} |   {stat[item]}")

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
            for sign in ["+", "-"]:
                print(f"> CHOICES DISTRIBUTION [{sign}]")
                mask = np.logical_and(self.label == item,
                                      self.plus_minus == sign)
                stat = Counter(self.preds_key[mask])
                print(f"{'ANSWERS':<{l}} | Count")
                for choice in self.mpi_choice_lst[sign]:
                    print(f"{choice:<{l}} |   {stat[choice]}")
                print("")
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
                print(
                    f"QUESTION #{idx+1:<4} | TRAIT: {self.label[idx]} | KEY: {self.plus_minus[idx]} | SCORE: {self.scores[idx]} | ANSWER: {self.preds_key[idx]}")
                print(f"> Statement: {statement}")
                print(
                    f"> Inverse Log-Perplexity: {list(np.round(np.array(self.likelihood[idx]), 4))}")
            line()
            f.close()
            sys.stdout = original_stdout

    def save_checkpoint(self, filename):
        torch.save(self, filename)


if __name__ == '__main__':
    ic("BERT prompting experiments on MPI dataset...")
    # Filename (may vary on your local computer)
    filename = "mpi_small"
    local_path = "Dataset/" + f"{filename}.csv"
    read_mpi(local_path, True, 5)
