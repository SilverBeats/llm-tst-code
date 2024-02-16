import abc
import math
from typing import Union

import fasttext
import pandas as pd
from pandas import DataFrame
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .utils import calc_bert_score, calc_corpus_bleu, eval_acc_by_fasttext, eval_model_loss


class BaseEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self):
        raise NotImplementedError


class HasOutputEvaluator(BaseEvaluator):
    def __init__(
            self,
            data_file_path: Union[str, DataFrame],
            acc_model_dir_or_path: str,
            d_ppl_model_dir: str,
            g_ppl_model_dir: str,
            bert_score_model_dir: str,
            bert_score_model_layers: int,
            ref_file_path: Union[str, DataFrame] = None,
            device: str = 'cuda:0',
            batch_size: int = 8,
            **kwargs
    ):
        super().__init__()
        print('load data from {}'.format(data_file_path))
        if isinstance(data_file_path, str):
            self.df = pd.read_csv(data_file_path).dropna().sort_values(by='id')
        else:
            self.df = data_file_path
        self.df['target'] = self.df['target'].apply(str.lower)
        self.df['source'] = self.df['source'].apply(str.lower)

        self.has_ref = False
        if ref_file_path is not None:
            self.has_ref = True
            if isinstance(ref_file_path, str):
                self.ref_df = pd.read_csv(ref_file_path).dropna().sort_values(by='id')
            else:
                self.ref_df = ref_file_path
            print('load ref-data from {}'.format(ref_file_path))
            self.ref_df = pd.read_csv(ref_file_path).sort_values(by='id')
            self.ref_df = self.ref_df[self.ref_df['id'].isin(self.df['id'])]
            self.ref_df['target'] = self.ref_df['target'].apply(str.lower)

        print(f'load classifier model from {acc_model_dir_or_path}')
        self.dis_model = fasttext.load_model(acc_model_dir_or_path)

        print('create d ppl model and tokenizer (gpt2 small)')
        self.d_ppl_tokenizer = GPT2Tokenizer.from_pretrained(d_ppl_model_dir)
        self.d_ppl_tokenizer.pad_token_id = self.d_ppl_tokenizer.eos_token_id
        self.d_ppl_tokenizer.pad_token = self.d_ppl_tokenizer.eos_token
        self.d_ppl_model = GPT2LMHeadModel.from_pretrained(d_ppl_model_dir).to(device).eval()

        print('create g ppl model and tokenizer (gpt2 large)')
        self.g_ppl_tokenizer = GPT2Tokenizer.from_pretrained(g_ppl_model_dir)
        self.g_ppl_tokenizer.pad_token_id = self.g_ppl_tokenizer.eos_token_id
        self.g_ppl_tokenizer.pad_token = self.g_ppl_tokenizer.eos_token
        self.g_ppl_model = GPT2LMHeadModel.from_pretrained(g_ppl_model_dir).to(device).eval()

        self.bert_score_config = {
            'model_type': bert_score_model_dir,
            'num_layers': bert_score_model_layers,
            'batch_size': batch_size,
            'lang': 'en',
            'device': device
        }

        self.batch_size = batch_size
        self.device = device

    def evaluate(self):
        source_texts = self.df['source'].tolist()
        target_texts = self.df['target'].tolist()
        ground_labels = self.df['t_label'].tolist()

        print('calculate self bleu')
        self_bleu = calc_corpus_bleu(source_texts, target_texts)

        if self.has_ref:
            ref_texts = self.ref_df['target'].tolist()
            print('calculate reference/human bleu')
            ref_bleu = calc_corpus_bleu(ref_texts, target_texts)
            print('calculate gm bleu')
            gm_bleu = math.pow(self_bleu * ref_bleu, 0.5)

        print('calculate d_ppl')
        d_ppl = eval_model_loss(self.d_ppl_model, self.d_ppl_tokenizer, target_texts, self.device, self.batch_size)
        print(d_ppl)
        print('calculate g_ppl')
        g_ppl = eval_model_loss(self.g_ppl_model, self.g_ppl_tokenizer, target_texts, self.device, self.batch_size)
        print('calculate t_ppl')
        gm_ppl = math.pow(d_ppl * g_ppl, 0.5)

        print('calculate acc')
        dis_acc = eval_acc_by_fasttext(target_texts, ground_labels, self.dis_model)

        if self.has_ref:
            print('calculate bert score')
            bert_score_result = calc_bert_score(ref_texts, target_texts, self.bert_score_config)

        return_dict = {
            'self_bleu': self_bleu,
            'acc': dis_acc,
            'd_ppl': d_ppl,
            'g_ppl': g_ppl,
            'gm_ppl': gm_ppl,
        }
        if self.has_ref:
            return_dict.update({
                'ref_bleu': ref_bleu,
                'gm_bleu': gm_bleu,
                **bert_score_result,
            })
        return return_dict
