import numpy as np
from icecream import ic
# HuggingFace & Torch
from transformers import BertModel, BertTokenizer, BertForMaskedLM, \
    BertForNextSentencePrediction, BertForQuestionAnswering
from torch.nn import functional as F
import torch


class BERT():
    def __init__(self, bert_version="bert-base-uncased"):
        self.bert_version = bert_version
        # TODO: (Xiaoyang) Add assertion check
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_version)
        # Masked LM
        self.MLM = BertForMaskedLM.from_pretrained(
            self.bert_version, return_dict=True)

    def transform(self, input):
        return self.bert_tokenizer.encode_plus(input, return_tensors='pt')

    def formatting_text(self, text):
        return text.replace('[MASK]', self.bert_tokenizer.mask_token)

    # TODO: (Xiaoyang) Enable multiple [MASK] prompting
    def predict_mlm(self, text, top_k=None):
        text = self.formatting_text(text)
        input = self.transform(text)
        mask_index = torch.where(
            input["input_ids"][0] == self.bert_tokenizer.mask_token_id)
        mask_word = F.softmax(self.MLM(**input).logits,
                              dim=-1)[0, mask_index, :]
        if top_k is not None:
            top_k = torch.topk(mask_word, top_k, dim=1)[1][0]
            for idx, token in enumerate(top_k):
                word = self.bert_tokenizer.decode([token])
                predicted_text = text.replace(
                    self.bert_tokenizer.mask_token, word)
                ic(f"{idx} | {predicted_text}")
        else:
            token = torch.argmax(mask_word, dim=1).squeeze()
            predicted_text = text.replace(
                self.bert_tokenizer.mask_token, self.bert_tokenizer.decode([token]))
            ic(predicted_text)
        return predicted_text


if __name__ == '__main__':
    ic("BERT Prompting Code...")
    bert = BERT()
    bert.predict_mlm("Paris is the [MASK] of France.")
    bert.predict_mlm("Paris is [MASK] capital of France.")
    # Something more interesting
    bert.predict_mlm("Who is Lebron James? An [MASK] player.")
    bert.predict_mlm("Who is [MASK] James? An NBA player.", top_k=10)
