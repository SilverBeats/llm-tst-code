import math
import os
from typing import Dict, Any
from typing import Union

import fasttext
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from constant import DATASET_TO_TST_TYPE, LLMType, Task, AccModelType, FluencyModelType
from utils import build_path
from .base import BaseEvaluator
from .utils import calc_bert_score, calc_corpus_bleu, eval_acc_by_fasttext, eval_model_loss, calc_acc_by_plm


class Evaluator(BaseEvaluator):
    def __init__(
            self,
            dataset: str = None,
            data_dir: str = None,
            output_dir: str = None,
            llm_type: LLMType = None,
            acc_model_type: AccModelType = None,
            acc_model_dir_or_path: str = None,
            fluency_model_type: FluencyModelType = None,
            d_ppl_model_dir: str = None,
            g_ppl_model_dir: str = 'gpt2-large',
            batch_size: int = 8,
            template_type: str = 'common',
            template_idx: int = 0,
            device: Union[torch.device, str] = 'cuda:0',
            bert_score_config: Dict[str, Any] = None,
            data_file_path: str = None,
            ref_file_path: str = None,
            **kwargs
    ):
        super().__init__()
        tst_type = DATASET_TO_TST_TYPE[dataset]
        if data_file_path is None:
            data_file_path = build_path(
                output_dir, template_type, template_idx,
                tst_type, dataset, str(Task.P), f'{llm_type.type}-processed.csv'
            )
        if ref_file_path is None:
            ref_file_path = build_path(data_dir, tst_type, dataset, 'reference.csv')
        if not os.path.exists(ref_file_path):
            ref_file_path = None

        print(f'Load data from {data_file_path}')
        self.df = pd.read_csv(data_file_path).dropna().sort_values(by='id')
        self.df['target'] = self.df['target'].apply(str.lower)
        self.df['source'] = self.df['source'].apply(str.lower)

        self.has_ref = ref_file_path is not None
        if self.has_ref:
            print(f'Load reference data from {ref_file_path}')
            self.ref_df = pd.read_csv(ref_file_path).dropna().sort_values(by='id')
            self.ref_df = pd.read_csv(ref_file_path).sort_values(by='id')
            self.ref_df = self.ref_df[self.ref_df['id'].isin(self.df['id'])]
            self.ref_df['target'] = self.ref_df['target'].apply(str.lower)

        print(f'Load classifier model from {acc_model_dir_or_path}')
        self.acc_model_type = acc_model_type
        if self.acc_model_type == AccModelType.FASTTEXT:
            self.dis_model = fasttext.load_model(acc_model_dir_or_path)
            self.dis_tokenizer = None
        elif self.acc_model_type == AccModelType.PLM:
            self.dis_model = AutoModelForSequenceClassification.from_pretrained(acc_model_dir_or_path).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(acc_model_dir_or_path)
        else:
            raise ValueError

        print('Load d-ppl model ...')
        self.fluency_model_type = fluency_model_type
        if self.fluency_model_type == FluencyModelType.PLM:
            self.d_ppl_tokenizer = AutoTokenizer.from_pretrained(d_ppl_model_dir)
            self.d_ppl_tokenizer.pad_token_id = self.d_ppl_tokenizer.eos_token_id
            self.d_ppl_tokenizer.pad_token = self.d_ppl_tokenizer.eos_token
            self.d_ppl_model = AutoModelForCausalLM.from_pretrained(d_ppl_model_dir).to(device).eval()
        else:
            raise ValueError

        print('Load g-ppl model')
        self.g_ppl_tokenizer = AutoTokenizer.from_pretrained(g_ppl_model_dir)
        self.g_ppl_tokenizer.pad_token_id = self.g_ppl_tokenizer.eos_token_id
        self.g_ppl_tokenizer.pad_token = self.g_ppl_tokenizer.eos_token
        self.g_ppl_model = AutoModelForCausalLM.from_pretrained(g_ppl_model_dir).to(device).eval()

        self.batch_size = batch_size
        self.device = device
        self.bert_score_config = bert_score_config

    def evaluate(self):
        source_texts = self.df['source'].tolist()
        target_texts = self.df['target'].tolist()
        ground_labels = self.df['t_label'].tolist()

        result_dict = {}
        print('calculate self bleu')
        result_dict['self_bleu'] = calc_corpus_bleu(source_texts, target_texts)

        if self.has_ref:
            ref_texts = self.ref_df['target'].tolist()
            print('calculate reference/human bleu')
            result_dict['ref_bleu'] = calc_corpus_bleu(ref_texts, target_texts)
            print('calculate gm bleu')
            result_dict['gm_bleu'] = math.pow(result_dict['self_bleu'] * result_dict['ref_bleu'], 0.5)
            print('calculate bert score')
            result_dict['bert_score_result'] = calc_bert_score(ref_texts, target_texts, self.bert_score_config)

        print('calculate d_ppl')
        if self.fluency_model_type == FluencyModelType.PLM:
            result_dict['d_ppl'] = eval_model_loss(
                self.d_ppl_model,
                self.d_ppl_tokenizer,
                target_texts,
                self.device,
                self.batch_size
            )
        print('calculate g_ppl')
        result_dict['g_ppl'] = eval_model_loss(
            self.g_ppl_model, self.g_ppl_tokenizer,
            target_texts, self.device, self.batch_size
        )
        print('calculate t_ppl')
        result_dict['gm_ppl'] = math.pow(result_dict['d_ppl'] * result_dict['g_ppl'], 0.5)

        print('calculate acc')
        if self.acc_model_type == AccModelType.FASTTEXT:
            result_dict['dis_acc'] = eval_acc_by_fasttext(target_texts, ground_labels, self.dis_model)
        elif self.acc_model_type == AccModelType.PLM:
            result_dict['dis_acc'] = calc_acc_by_plm(
                target_texts, ground_labels, self.dis_model,
                self.dis_tokenizer, self.batch_size, self.device
            )
        return result_dict
