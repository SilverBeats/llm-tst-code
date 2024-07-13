import json
import os
import random
import time
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

import pandas as pd
import requests
from tqdm import tqdm

from utils import build_path, convert_label_to_style
from .base import BaseRewriter


class OpenAIRewriter(BaseRewriter):
    API = 'https://api.songshuai.xyz/v1/chat/completions'

    def __init__(
            self,
            output_dir: str,
            llm_type: str,
            tst_type: str,
            data_file_path: str,
            api_keys: List[str],
            template: str,
            **kwargs
    ):
        super().__init__()
        # os.environ['http_proxy'] = 'http://localhost:7890'
        # os.environ['https_proxy'] = 'http://localhost:7890'
        if api_keys is None:
            api_keys = []
        assert len(api_keys) > 0
        self.llm_type = llm_type
        self.output_dir = output_dir
        self.data_file_path = data_file_path
        self.template = template
        self.api_keys = api_keys
        self.tst_type = tst_type

    def _process_input_text(self, raw_text: str, label: int):
        if 'caption' in self.data_file_path.lower():
            source_style = 'factual'
            target_style = convert_label_to_style(label, self.tst_type)
        else:
            source_style = convert_label_to_style(label, self.tst_type)
            target_style = convert_label_to_style(1 - label, self.tst_type)
        return self.template.format(source_style, target_style, raw_text)

    def _chat_gpt(self, _id: int, raw_text: str, label: int):
        try:
            time.sleep(random.randint(1, 15))
            api_key = self.api_keys[_id % len(self.api_keys)]

            response = requests.post(
                url=self.API,
                json={
                    'model': self.llm_type,
                    'messages': [
                        {
                            'role': 'user',
                            'content': self._process_input_text(raw_text, label)
                        }
                    ]
                },
                headers={
                    'Authorization': api_key,
                    'Content-Type': 'application/json'
                },
                timeout=20
            )
            completion = json.loads(response.content)
            return {
                'process': {
                    'id': _id,
                    'source': raw_text,
                    's_label': 2 if 'caption' in self.data_file_path.lower() else label,
                    'target': completion['choices'][0]['message']['content'],
                    't_label': label if 'caption' in self.data_file_path.lower() else 1 - label
                },
                'input_tokens': completion['usage']['prompt_tokens'],
                'output_tokens': completion['usage']['completion_tokens'],
            }
        except Exception as e:
            return {
                'process': {
                    'id': _id,
                    'source': raw_text,
                    's_label': 2 if 'caption' in self.data_file_path.lower() else label,
                    'target': 'An error occurred: {}'.format(str(e)),
                    't_label': label if 'caption' in self.data_file_path.lower() else 1 - label
                },
                'input_tokens': 0,
                'output_tokens': 0
            }

    def rewrite(self):
        success_output_file_path = build_path(self.output_dir, f'{self.llm_type}.csv')
        error_output_file_path = build_path(self.output_dir, f'{self.llm_type}-error.csv')

        print('load dataset from {}'.format(self.data_file_path))
        data_df = pd.read_csv(self.data_file_path)
        print(f'data size = {data_df.shape[0]}')

        success, error = [], []
        input_tokens = output_tokens = 0

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            label_key = 'label' if 'caption' not in self.data_file_path else 't_label'
            futures = [
                pool.submit(self._chat_gpt, row['id'], row['text'], row[label_key])
                for i, row in data_df.iterrows()
            ]
            pbar = tqdm(total=len(futures), dynamic_ncols=True)
            for future in as_completed(futures):
                return_dict = future.result()
                if return_dict['input_tokens'] == 0:
                    error.append(return_dict['process'])
                else:
                    success.append(return_dict['process'])
                input_tokens += return_dict['input_tokens']
                output_tokens += return_dict['output_tokens']
                pbar.update(1)
                pbar.set_postfix({'input_tokens': input_tokens, 'output_tokens': output_tokens})

        if len(success) > 0:
            pd.DataFrame(success).sort_values(by='id').to_csv(success_output_file_path, index=False)
            print(f'saved at {success_output_file_path}')

        if len(error) > 0:
            pd.DataFrame(error).sort_values(by='id').to_csv(error_output_file_path, index=False)
            print(f'saved at {error_output_file_path}')
