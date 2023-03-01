from email.policy import default
from functools import reduce
from bert_prompt import *
import pandas as pd
from collections import Counter, defaultdict
from utils import *
from tqdm import tqdm
import sys
from tabulate import tabulate


OCEAN = ['O', 'C', 'E', 'A', 'N']
MPI_CHOICES = ['(A)', '(B)', '(C)', '(D)', '(E)']
MPI_CHOICES_DESC = [
    "Very Accurate",
    "Moderately Accurate",
    "Neither Accurate Nor Inaccurate",
    "Moderately Inaccurate",
    "Very Inaccurate"]
MPI_CHOICE_ALL = reduce(
    lambda lst, z: lst + [z[0] + " " + z[1]], zip(MPI_CHOICES, MPI_CHOICES_DESC), [])


def read_mpi(path, show=False, n=None):
    df = pd.read_csv(path)
    if show:
        assert n is not None, 'Please specify how many rows you want to view.'
        ic(df.head(n))
    print(f"Personality Test on MPI dataset...")
    print(f"There are {len(df)} multiple choice questions in total.")
    return df


def prepare_mpi_questions(statement):
    # TODO:(Xiaoyang) change this template if necessary
    questions = f"Given a statement of you: \"You {statement}.\" "
    prompt = "Please choose from the following options to identify how accurately this statement describes you."
    # options = "\nOptions: \n(B). Moderately Accurate \n(A). Very Accurate \n(C). Neither Accurate Nor Inaccurate \n(D). Moderately Inaccurate \n(E). Very Inaccurate \nAnswers: "
    options = "\nOptions: \n(A). Very Accurate \n(B). Moderately Accurate \n(C). Neither Accurate Nor Inaccurate \n(D). Moderately Inaccurate \n(E). Very Inaccurate \nAnswers: "
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
def logit_to_prob(logit):
    # logits: L x Vocab_Size
    prob = torch.softmax(logit, dim=-1)
    return torch.max(prob, dim=-1)[0]


def prob_to_ll(prob, ll_type, choice_len):
    if ll_type == 'mean-a':
        return torch.mean(torch.log(prob)[-choice_len:])
    elif ll_type == 'mean-s':
        return torch.mean(torch.log(prob))
    else:
        assert False, 'Unrecognized input argument.'


def run_mpi(path_to_dset, start, end,
            model, tokenizer, version,
            mpi_choice, ll_type,
            filename):
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
        print(f'MODEL: BERT | Version: {version}')
        mpi = MPI(path_to_dset, start, end, mpi_choice)
        mpi.reset()
        mpi.run(tokenizer, model, ll_type=ll_type)
        mpi.display_ocean_stats()
        mpi.display_aux_stats()
        f.close()
        sys.stdout = original_stdout
        return mpi


class MPI():
    def __init__(self, path_to_file, start, end, mpi_choice=MPI_CHOICE_ALL):
        self.mpi_df = read_mpi(path_to_file)
        # (Optional): only testing the first few examples
        self.mpi_df = self.mpi_df[start: end]
        # STATEMENT
        self.text = np.array(self.mpi_df['text'])
        # QUESTIONS
        self.questions = np.array([prepare_mpi_questions(x)
                                  for x in self.text])
        # ic(self.questions.shape)
        # TODO:(Xiaoyang) Enable argument passing later...
        self.mpi_choice_lst = mpi_choice
        # self.mpi_choice_lst = MPI_CHOICES
        # LABEL
        self.label = np.array(self.mpi_df['label_ocean'])
        # KEY
        self.key = torch.tensor(self.mpi_df['key'], dtype=torch.long)
        self.plus_minus = ["+" if k == 1 else "-" for k in self.key]
        # OCEAN SCORE
        self.OCEAN = defaultdict(list)
        self.raw_scores,  self.scores = [], []
        # META-DATA
        self.likelihood, self.probs = [], []
        self.preds_key, self.preds = [], []
        # DEAL WITH FILE WRITE

        # SANITY CHECK CODE (Optional)
        # check_column_cleanness(self.mpi_df, 'label_ocean')
        # check_column_cleanness(self.mpi_df, 'key')

    def reset(self):
        self.OCEAN = defaultdict(list)
        self.raw_scores, self.scores = [], []
        self.probs, self.likelihood = [], []
        self.preds_key, self.preds = [], []
        # TODO: (Xiaoyang) more functionality here...

    def run(self, tokenizer, model, ll_type="mean-a"):

        print("--------------------------------------")
        print(f"Sample questions look like this:")
        print(f"{self.questions[0]}")
        print("--------------------------------------")
        print("MCQA task starts...")
        print("--------------------------------------")
        for idx, prompt in enumerate(tqdm(self.questions)):
            ll_lst, prob_lst = [], []
            # NOTE: currently unequal sequence length forbids us doing batch-level operation
            # TODO: (Xiaoyang) Find a way to do batch-level processing later...
            for choice in self.mpi_choice_lst:
                # print((prompt + choice)[-len(choice):])
                tokens = tokenizer(
                    prompt + choice, return_tensors="pt", padding=True)
                out = model(**tokens)
                logit = out.logits
                # LOG-LIKELIHOOD CALCULATION
                prob = logit_to_prob(logit.squeeze())
                ll = prob_to_ll(prob, ll_type, len(choice))
                ll_lst.append(ll.item())
                # PROBABILITY FOR EACH WORD IN THE SENTENCE
                prob_lst.append(prob)
            # MCQA BASED ON LIKELIHOOD
            ll_lst = torch.tensor(ll_lst)
            pred = torch.argmax(ll_lst).item()
            # ic(pred)
            print(
                f"Question #{idx:<4} | Trait: {self.label[idx]} | Key: {self.plus_minus[idx]} | ANSWER: {self.mpi_choice_lst[pred]}")
            # print(f"-- ANSWER: {MPI_IDX_TO_KEY[pred]}")
            print(f"-- Likelihood: {list(np.round(np.array(ll_lst), 4))}")
            # SAVE STATISTICS
            self.likelihood.append(ll_lst)
            self.probs.append(prob_lst)
            # SAVE MCQA-ANSWERS
            self.preds_key.append(self.mpi_choice_lst[pred])
            self.preds.append(pred)
            self.raw_scores.append(MPI_IDX_TO_SCORE[pred])
        # SCORE CALCULATION
        self.calculate_score()

    def calculate_score(self):
        self.scores = torch.tensor(self.raw_scores) * \
            self.key[0:len(self.raw_scores)]
        for idx, score in enumerate(self.scores):
            self.OCEAN[self.label[idx]].append(score)

    def display_ocean_stats(self):
        # ic(self.OCEAN)
        print("--------------------------------------")
        print("OCEAN SCORES STATS")
        self.stats = {}
        for item in OCEAN:
            vals = torch.tensor(self.OCEAN[item], dtype=torch.float32)
            mean, std = torch.mean(vals).item(), torch.std(vals).item()
            self.stats[item] = [mean, std]
            print(
                f"{item} | MEAN: {mean} | STD: {std}")
        # TODO: (Xiaoyang) add more functionality here
        # self.reset()
        # return np.array(self.stats)

    def display_aux_stats(self):
        print("--------------------------------------")
        print("OTHER INTERESTING STATS")
        # Format length
        l = max(7, max([len(x) for x in self.mpi_choice_lst]))
        stat = Counter(self.preds_key)
        print(f"{'ANSWERS':<{l}} | Count")
        for item in self.mpi_choice_lst:
            print(f"{item:<{l}} |   {stat[item]}")

    def display_trait_stats(self):
        print("--------------------------------------")
        print("TRAITS-LEVEL STATS")
        score = list(np.arange(-5, 0, 1)) + list(np.arange(1, 6, 1))
        for item in OCEAN:
            print(f"Trait: {item}")
            count = dict(Counter(np.array(self.OCEAN[item])))
            df = pd.DataFrame()
            df[score] = None
            # TODO:(Xiaoyang) revision later...
            df.loc[len(df.index)] = [
                count[x] if x in count else x for x in score]
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))


if __name__ == '__main__':
    ic("BERT prompting experiments on MPI dataset...")
    # Filename (may vary on your local computer)
    filename = "mpi_small"
    local_path = "Dataset/" + f"{filename}.csv"
    read_mpi(local_path, True, 5)

    # Declare MPI instance
    mpi = MPI(local_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
    # model = BertForMultipleChoice.from_pretrained("bert-base-uncased")
    model = BertForMultipleChoice.from_pretrained("bert-large-cased")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # model = BertForMultipleChoice.from_pretrained("bert-base-uncased")
    mpi.run(tokenizer, model)
    mpi.display_score()
