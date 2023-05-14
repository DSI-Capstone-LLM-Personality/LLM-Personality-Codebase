
# ----------------------------- #
# SCRIPT FOR BERT MODEL SETUP   #
# AUTHOR: XIAOYANG SONG         #
# ----------------------------- #
import torch
import openai
from torch.nn import functional as F
import numpy as np
from util.utils import *
from icecream import ic
# HuggingFace & Torch
from transformers import AlbertForPreTraining, AutoTokenizer, GPT2LMHeadModel, \
    GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoXForCausalLM, GPTNeoXTokenizerFast,\
    OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, pipeline, BertLMHeadModel, RobertaForCausalLM, \
    BartForConditionalGeneration, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, OPTForCausalLM

# openai.api_key = read_api_key("", 'xysong')
# Link to OPT: https://huggingface.co/models?search=opt
# Link to available models: https://huggingface.co/transformers/v2.4.0/pretrained_models.html
# Link to generate(): https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/text_generation#transformers.GenerationMixin.generate
# Link to GPT generation: https://www.kaggle.com/code/tuckerarrants/text-generation-with-huggingface-gpt2
########## CONSTRAINT MCQA UTILITIES  ##########
#---------- Language Models ----------#
MODEL = {
    'Constraint': {
        'BERT': BertLMHeadModel,
        'RoBERTa': RobertaForCausalLM,
        'ALBERT': AlbertForPreTraining,
        'SpanBERT': None,
        'GPT': OpenAIGPTLMHeadModel,
        'GPT2': GPT2LMHeadModel,
        'GPTNEO': GPTNeoForCausalLM,
        'GPTNEOX': GPTNeoXForCausalLM,
        'BART': BartForConditionalGeneration,
        'T5': T5ForConditionalGeneration,
        'FLAN-T5': T5ForConditionalGeneration,
        'OPT': OPTForCausalLM
    },
    'Open-Vocab': {
        'GPT2': GPT2LMHeadModel,
        'GPTNEO': GPTNeoForCausalLM,
        'BART': BartForConditionalGeneration,
        'T5': T5ForConditionalGeneration,
        'FLAN-T5': T5ForConditionalGeneration,
        'T0': AutoModelForSeq2SeqLM
    }
}
TOKENIZER = {'BERT': AutoTokenizer,
             'ALBERT': AutoTokenizer,
             'RoBERTa': AutoTokenizer,
             'GPT': OpenAIGPTTokenizer,
             'GPT2': GPT2Tokenizer,
             'GPTNEO': GPT2Tokenizer,
             'GPTNEOX': GPTNeoXTokenizerFast,
             'BART': AutoTokenizer,
             'T5': AutoTokenizer,
             'FLAN-T5': AutoTokenizer,
             'T0': AutoTokenizer,
             'OPT': AutoTokenizer}
#---------- Language model Perplexity  ----------#


def logit_to_prob(logit, ids):
    # logits: L x Vocab_Size
    # ic(logit.shape)
    # ic(ids.shape)
    assert logit.shape[0] == ids.shape[0]
    prob = torch.softmax(logit, dim=-1)
    # ic(prob.shape)
    # ic(ids.shape)
    return prob[np.arange(ids.shape[0]), ids]


def find_critical_word(tokenizer, encoded_seq):
    '''gives the tokenization of the last word in the sequence'''

    critical_word = ''
    tokens_indices = []  # tokenized indices of the sequence

    for token in torch.flip(encoded_seq[0], dims=(0,)):
        token_string = tokenizer.convert_ids_to_tokens([token])[0]
        tokens_indices.insert(0, token)
        if 'Ġ' in token_string:
            critical_word = token_string[1:] + critical_word
            break
        else:
            critical_word = token_string + critical_word

    return critical_word, tokens_indices


def find_critical_phrase(tokenizer, encoded_seq, target):
    target = set(target.split())
    phrase, toi = set(), []
    seq = encoded_seq
    while phrase != target:
        # ic(seq)
        word, tokens = find_critical_word(tokenizer, seq)

        # Update
        seq = seq[:, :-len(tokens)]
        if seq.shape[1] == 0:
            break
        phrase.add(word)
        toi = tokens + toi
        # Check
        if phrase == target:
            return toi
    assert False, 'There is something wrong in tokenization.'


def prob_to_ll(prob): return torch.mean(torch.log(prob))


class LMPROB():
    def __init__(self, family, model, tokenizer, ll_type, half_precision):
        self.family = family
        self.model = model
        self.tokenizer = tokenizer
        self.ll_type = ll_type
        self.half_precision = half_precision


    def __call__(self, prompt, choice, tmp):
        # ic(prompt)
        # ic(choice)
        tokens = self.tokenizer(
            prompt.strip() + " " + choice, return_tensors="pt", add_special_tokens=False)
        # print(prompt + choice)
        # Calibration
        # print(tmp)
        tmp = tmp.strip() + " " + choice
        tmp_tokens = self.tokenizer(tmp, return_tensors="pt", add_special_tokens=False)
        
        # Not used: normalization
        # answer = "Answer: " + choice
        # ans_tokens = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False)

        for item, _ in tokens.items():
            tokens[item] = tokens[item].to(DEVICE)

        if self.family in ['GPT2', 'GPTNEO', 'GPTNEOX', 'OPT']:
            # token sequence processing
            logits = self.model(**tokens).logits
            # Not used: normalization method
            # ans_logits = self.model(**ans_tokens).logits
            # ic(logits.shape)
            tmp_logits = self.model(**tmp_tokens).logits

            if self.ll_type == 'ans_inv_perp':
                # ic(choice)
                toi = find_critical_phrase(
                    self.tokenizer, tokens.input_ids, choice)
                
                # ans_toi = find_critical_phrase(
                #     self.tokenizer, ans_tokens.input_ids, choice)
                
                tmp_toi = find_critical_phrase(
                    self.tokenizer, tmp_tokens.input_ids, choice)

            elif self.ll_type == 'sent_inv_perp':
                toi = tokens.input_ids[0]
            else:
                assert False, 'Unrecognized input argument.'

            # calibration
            prob = torch.softmax(logits.squeeze(), dim=-1)[-len(toi):]
            tmp_prob = torch.softmax(tmp_logits.squeeze(), dim=-1)[-len(tmp_toi):]
            # ic(prob.shape)
            # ic(tmp_prob.shape)
            probs = prob / tmp_prob
            # ic(probs)
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)
            # ic(probs)
            # ic(probs.shape)
            ids = tokens.input_ids[0][-len(tmp_toi):]
            probs = probs[np.arange(ids.shape[0]), ids]
            # ic(probs.shape)
            # extract probability and inverse perplexity
            # probs = logit_to_prob(
            #     logits.squeeze(), tokens.input_ids[0])[-len(toi):]
            # Not used: normalization
            # ans_probs = logit_to_prob(ans_logits.squeeze(), ans_tokens.input_ids[0])[-len(ans_toi):]
            
            tmp_probs = logit_to_prob(tmp_logits.squeeze(), tmp_tokens.input_ids[0])[-len(tmp_toi):]
            # PREVIOUS
            ll = prob_to_ll(probs) 
            # ic("HELLO WORLD")
            # This is the code for normalization (not used)
            # ll = torch.sum(torch.log(probs)) - torch.sum(torch.log(tmp_probs))
            # ic(ll)
            # ic(np.exp(-ll))
            # ic(torch.prod(probs)**(-1 / len(probs)))
            return probs, ll, toi
        if self.family in ['BERT', 'RoBERTa', 'ALBERT', 'BART', 'T5', 'GPT']:
            raise NotImplementedError
        else:
            assert False, 'Unrecognized model family'

########## OPEN VOCAB MCQA UTILITIES  ##########
#---------- Language Models  ----------#


class PROMPTER():
    def __init__(self, family, model, tokenizer, g_config, version=None):
        self.family, self.version = family, version
        self.model, self.tokenizer = model, tokenizer
        self.g_config = g_config
        assert 'top_p' in self.g_config
        assert 'temperature' in self.g_config
        assert 'max_tokens' in self.g_config

    def __call__(self, prompt):
        if self.family == 'GPT3':
            assert self.version is not None
            response = openai.Completion.create(
                model=self.version,
                prompt=prompt,
                temperature=self.g_config['temperature'],
                max_tokens=self.g_config['max_tokens'],
                top_p=self.g_config['top_p']
            )
            return response['choices'][0]['text'].strip()

        elif self.family in ["GPT2", "GPTNEO", "BART", "T0"]:
            # TODO: (Xiaoyang) Add paddings
            inputs = self.tokenizer(prompt, return_tensors='pt')
            input_ids = inputs.input_ids.to(DEVICE)
            response = self.model.generate(input_ids, do_sample=True,
                                           top_p=self.g_config['top_p'], top_k=0,
                                           temperature=self.g_config['temperature'],
                                           max_new_tokens=self.g_config['max_tokens'])
            output = self.tokenizer.decode(
                response[0][len(input_ids[0]):])  # only get new tokens
            print(len(response[0][len(input_ids[0]):]))
            # output = self.tokenizer.decode(response[0])
            return output
        elif self.family in ["FLAN-T5"]:
            inputs = self.tokenizer(prompt, return_tensors='pt')
            input_ids = inputs.input_ids.to(DEVICE)
            response = self.model.generate(input_ids,
                                           top_p=self.g_config['top_p'],
                                           temperature=self.g_config['temperature'],
                                           max_new_tokens=self.g_config['max_tokens'])
            output = self.tokenizer.decode(
                response[0][1:-1])  # remove <pad> and <\s>
            return output
        elif self.family in ["T5"]:
            # TODO: (Morris) Find reasonable reponse
            inputs = self.tokenizer(prompt, return_tensors='pt')
            input_ids = inputs.input_ids.to(DEVICE)
            response = self.model.generate(input_ids,
                                           top_p=self.g_config['top_p'],
                                           temperature=self.g_config['temperature'],
                                           max_new_tokens=self.g_config['max_tokens'])
            output = self.tokenizer.decode(response[0])
            return output
        else:
            assert False, 'Unrecognized model Type.'

