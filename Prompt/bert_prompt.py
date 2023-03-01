# ----------------------------- #
# SCRIPT FOR BERT MLM PROMPTING #
# AUTHOR: XIAOYANG SONG         #
# ----------------------------- #
import torch
from torch.nn import functional as F
import numpy as np
from icecream import ic
# HuggingFace & Torch
from transformers import AutoTokenizer, BertModel, BertTokenizer, BertForMaskedLM, AutoTokenizer,\
    BertForNextSentencePrediction, BertForQuestionAnswering, pipeline, BertForMultipleChoice, BertLMHeadModel


class BERT():
    def __init__(self, bert_version="bert-base-uncased"):
        self.bert_version = bert_version
        # TODO: (Xiaoyang) Add assertion check
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_version)
        # Masked LM
        self.MLM = BertForMaskedLM.from_pretrained(
            self.bert_version, return_dict=True)
        # MCQA
        self.mcqa = BertForMultipleChoice.from_pretrained(self.bert_version)
        # QA
        self.QA = BertForQuestionAnswering.from_pretrained(self.bert_version)
        # TODO: (Xiaoyang) add more task-specific BERT variants here...

    def transform(self, input):
        return self.bert_tokenizer.encode_plus(input, return_tensors='pt')

    def formatting_text(self, text):
        return text.replace('[MASK]', self.bert_tokenizer.mask_token)

    # TODO: (Xiaoyang) Enable MULTIPLE [MASK] prompting
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

    def predict_qa(self, question, context):
        # TODO: (Xiaoyang) Increase generalizability later
        inputs = self.bert_tokenizer.encode_plus(
            question, context, return_tensors='pt')
        # ic(inputs)
        output = self.QA(**inputs)
        start_max = output.start_logits.argmax()
        end_max = output.end_logits.argmax() + 1
        answer = self.bert_tokenizer.decode(
            inputs["input_ids"][0][start_max: end_max], skip_special_tokens=True)
        ic(answer)
        return answer


if __name__ == '__main__':
    ic("BERT Prompting Experiments...")
    # bert = BERT()
    # bert.predict_mlm("Paris is the [MASK] of France.")
    # bert.predict_mlm("Paris is [MASK] capital of France.")
    # Something more interesting
    # bert.predict_mlm("Who is Lebron James? An [MASK] player.")
    # bert.predict_mlm("Who is [MASK] James? An NBA player.", top_k=10)
    # bert.predict_mlm("Columbia University is at [MASK] city.")
    # bert.predict_mlm("[MASK] University is at New York city.")
    # QA
    # Trying out different pretrained LM
    # bert = BERT("deepset/bert-base-cased-squad2")
    # question, context = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    # bert.predict_qa(question, context)

    # question, context = "What is the capital of France?", "The capital of France is Paris."
    # bert.predict_qa(question, context)

    # question, context = "What is my name?", "Xiaoyang is my name."
    # bert.predict_qa(question, context)

    # question, context = "What is my name?", "My name is Xiaoyang."
    # bert.predict_qa(question, context)
    # bert.predict_qa(mpi_template+choice, "ABCDE")

    # TODO: (Xiaoyang) Check out source code
    # unmasker = pipeline('fill-mask', model='xlm-roberta-large')
    # print(unmasker(mpi_template+choice))
