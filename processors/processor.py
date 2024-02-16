import os
from typing import Callable, Dict

import pandas as pd
from tqdm import tqdm

from global_config import DATASET_TO_TST_TYPE, LLMType, Task
from processors.utils import (process_7b_llama2, process_gpt, process_mistral,
                              process_qwen_14b_chat, process_qwen_7b_chat, interval_abbr)
from utils import build_path

LLM_TO_PROCESS_FUN: Dict[str, Callable] = {
    LLMType.qwen_7b_chat.type: process_qwen_7b_chat,
    LLMType.qwen_14b_chat.type: process_qwen_14b_chat,
    LLMType.gpt_3_5_turbo.type: process_gpt,
    LLMType.gpt_4.type: process_gpt,
    LLMType.mistral_7b_instruct.type: process_mistral,
    LLMType.llama2_7b_chat_hf.type: process_7b_llama2,
    LLMType.llama2_13b_chat_hf.type: process_7b_llama2,
}


class Processor:

    def __init__(
            self,
            dataset: str,
            llm_type: LLMType,
            output_dir: str = 'output',
            template_type: str = 'common',
            template_idx: int = 0,
            **kwargs
    ):
        tst_type = DATASET_TO_TST_TYPE[dataset]
        data_dir = build_path(
            output_dir,  template_type, template_idx,
            tst_type, dataset, str(Task.R)
        )
        output_dir = build_path(
            output_dir, template_type, template_idx,
            tst_type, dataset, str(Task.P)
        )
        os.makedirs(output_dir, exist_ok=True)
        self.dataset = dataset
        self.rewrite_file_path = build_path(data_dir,  f'{llm_type.type}.csv')
        self.output_file_path = build_path(output_dir, f'{llm_type.type}-processed.csv')
        self.human_process_file_path = build_path(output_dir, f'{llm_type.type}-processed-human.csv')
        self.fun = LLM_TO_PROCESS_FUN[llm_type.type]
        if self.fun is None:
            raise NotImplementedError

    def process(self):
        success, error = [], []

        data_list = pd.read_csv(self.rewrite_file_path).to_dict('records')
        for row in tqdm(data_list, total=len(data_list), desc='processing ...'):
            processed_result = self.fun(row['target'])
            if processed_result == -1:
                error.append(row)
            else:
                row['target'] = processed_result
                success.append(row)
        if len(success) > 0:
            success_df = pd.DataFrame(success).sort_values(by='id')
            success_df['source'] = success_df['source'].apply(str.lower)
            success_df['target'] = success_df['target'].apply(str.lower)
            if self.dataset in ('imdb', 'gender', 'political', 'yelp', 'imagecaption'):
                success_df['target'] = success_df['target'].apply(interval_abbr, args=(1,))
            elif self.dataset in ('amazon', ):
                success_df['target'] = success_df['target'].apply(interval_abbr, args=(2,))
            success_df.to_csv(self.output_file_path, index=False)
            print('saved at {}'.format(self.output_file_path))

        if len(error) > 0:
            error_df = pd.DataFrame(error).sort_values(by='id')
            error_df['source'] = error_df['source'].apply(str.lower)
            error_df['target'] = error_df['target'].apply(str.lower)
            if self.dataset in ('imdb', 'gender', 'political', 'yelp', 'imagecaption'):
                error_df['target'] = error_df['target'].apply(interval_abbr, args=(1,))
            elif self.dataset in ('amazon',):
                error_df['target'] = error_df['target'].apply(interval_abbr, args=(2,))
            error_df.to_csv(self.human_process_file_path, index=False)
            print(f'A total of {len(error)} pieces of data need to be manually repaired.')
            print(f'Save file to {self.human_process_file_path}')
