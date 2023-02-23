from bert_prompt import *
import pandas as pd
from collections import Counter, defaultdict
from utils import *
from tqdm import tqdm


def read_mpi(path, show=False, n=None):
    df = pd.read_csv(path)
    if show:
        assert n is not None, 'Please specify how many rows you want to view.'
        ic(df.head(n))
    print(f"There are {len(df)} multiple choice questions in total.")
    return df


def prepare_mpi_questions(statement):
    questions = f"Given a statement of you: 'You {statement}'. Please choose from the following options to identify how accurately this statement describes you."
    options = " Options: (A). Very Accurate (B). Moderately Accurate (C). Neither Accurate Nor Inaccurate (D). Moderately Inaccurate (E). Very Inaccurate Answers: "
    return questions + options


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
    def __init__(self, path_to_file):
        self.mpi_df = read_mpi(path_to_file)
        # STATEMENT
        self.text = np.array(self.mpi_df['text'])
        # QUESTIONS
        self.questions = np.array([prepare_mpi_questions(x)
                                  for x in self.text])
        self.mpi_choice_lst = ['A', 'B', 'C', 'D', 'E']
        # LABEL
        self.label = np.array(self.mpi_df['label_ocean'])
        # KEY
        self.key = torch.tensor(self.mpi_df['key'], dtype=torch.long)
        # OCEAN score dict
        self.OCEAN = defaultdict(list)
        # Metadata
        self.logits, self.probs = [], []
        self.preds_key, self.preds, self.scores = [], [], []

        # Sanity check code (optional)
        check_column_cleanness(self.mpi_df, 'label_ocean')
        check_column_cleanness(self.mpi_df, 'key')

    def reset(self):
        self.OCEAN = defaultdict(list)
        self.probs, self.logits = [], []
        self.preds_key, self.preds = [], []
        # TODO: (Xiaoyang) more functionality here...

    def run(self, tokenizer, model):
        for prompt in tqdm(self.questions):
            tokens = tokenizer([prompt] * len(self.mpi_choice_lst), self.mpi_choice_lst,
                               return_tensors="pt", padding=True)
            out = model(**{k: v.unsqueeze(0) for k, v in tokens.items()})
            logit = out.logits
            pred = torch.argmax(logit)
            # Store results
            self.logits.append(logit)
            self.probs.append(torch.softmax(logit, -1))
            self.preds_key.append(MPI_IDX_TO_KEY[pred])
            self.preds.append(pred)
            self.scores.append(MPI_IDX_TO_SCORE[pred])
        # Calculating scores
        self.scores = torch.tensor(self.scores) * self.key
        ic(self.scores)
        # Store scores into OCEAN
        for idx, score in enumerate(self.scores):
            self.OCEAN[self.label[idx]].append(score)


if __name__ == '__main__':
    ic("BERT prompting experiments on MPI dataset...")
    # Filename (may vary on your local computer)
    filename = "mpi_small"
    local_path = "Dataset/" + f"{filename}.csv"
    read_mpi(local_path, True, 5)

    # Declare MPI instance
    mpi = MPI(local_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMultipleChoice.from_pretrained("bert-base-uncased")
    mpi.run(tokenizer, model)