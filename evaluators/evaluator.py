import json
import os
from pprint import pp
from typing import Union

import torch

from global_config import DATASET_TO_TST_TYPE, LLMType, Task
from utils import build_path
from .amazon import AmazonEvaluator
from .caption import CaptionEvaluator
from .gender import GenderEvaluator
from .political import PoliticalEvaluator
from .yelp import YelpEvaluator

DATASET_TO_EVALUATOR = {
    'amazon': AmazonEvaluator,
    'yelp': YelpEvaluator,
    'gender': GenderEvaluator,
    'political': PoliticalEvaluator,
    'imagecaption': CaptionEvaluator,
}


class Evaluator:
    def __init__(
            self,
            dataset: str,
            data_dir: str,
            output_dir: str,
            llm_type: LLMType,
            acc_model_dir_or_path: str,
            batch_size: int = 8,
            template_type: str = 'common',
            template_idx: int = 0,
            device: Union[torch.device, str] = 'cuda:0',
            d_ppl_model_dir: str = 'gpt2',
            g_ppl_model_dir: str = 'gpt2-large',
            bert_score_model_dir: str = 'roberta-large',
            bert_score_model_layers: int = 17,
            **kwargs
    ):
        tst_type = DATASET_TO_TST_TYPE[dataset]

        data_file_path = build_path(
            output_dir, template_type, template_idx,
            tst_type, dataset, str(Task.P), f'{llm_type.type}-processed.csv'
        )
        ref_file_path = build_path(data_dir, tst_type, dataset, 'reference.csv')

        eval_result_dir = build_path(
            output_dir, template_type, template_idx,
            tst_type, dataset, str(Task.E)
        )
        os.makedirs(eval_result_dir, exist_ok=True)
        self.eval_result_file_path = build_path(eval_result_dir, f'{llm_type.type}-eval.json')

        args = {
            'data_file_path': data_file_path,
            'ref_file_path': ref_file_path,
            'batch_size': batch_size,
            'device': device,
            'acc_model_dir_or_path': acc_model_dir_or_path,
            'd_ppl_model_dir': d_ppl_model_dir,
            'g_ppl_model_dir': g_ppl_model_dir,
            'bert_score_model_dir': bert_score_model_dir,
            'bert_score_model_layers': bert_score_model_layers
        }
        self.evaluator = DATASET_TO_EVALUATOR[dataset](**args, **kwargs)

    def evaluate(self):
        result_dict = self.evaluator.evaluate()
        pp(result_dict)
        with open(self.eval_result_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)
        print('saved at {}'.format(self.eval_result_file_path))
