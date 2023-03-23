
# ----------------------------- #
# SCRIPT FOR BERT MODEL SETUP   #
# AUTHOR: XIAOYANG SONG         #
# ----------------------------- #
import torch
from torch.nn import functional as F
import numpy as np
from icecream import ic
# HuggingFace & Torch
from transformers import AutoTokenizer, BertModel, BertTokenizer, \
    BertForMaskedLM, AutoTokenizer, BertForNextSentencePrediction, \
    BertForQuestionAnswering, GPT2LMHeadModel, GPT2Tokenizer, pipeline, BertForMultipleChoice, \
    BertLMHeadModel, RobertaForCausalLM, AutoConfig

# https://huggingface.co/transformers/v2.4.0/pretrained_models.html
MODEL = {
    'BERT': BertLMHeadModel,
    'RoBERTa': RobertaForCausalLM,
    'SpanBERT': None,
    'GPT2': GPT2LMHeadModel}
TOKENIZER = {'BERT': AutoTokenizer,
             'RoBERTa': AutoTokenizer,
             'GPT2': GPT2Tokenizer}
