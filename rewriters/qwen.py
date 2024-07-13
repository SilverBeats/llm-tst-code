import random
import time
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any, Dict, List, Tuple

import pandas as pd
from dashscope import Generation
from tqdm import tqdm

from utils import build_path, convert_label_to_style
from .base import BaseRewriter


class QwenRewriter(BaseRewriter):
    ERROR_CODE = ['Throttling', 'Throttling.RateQuota', 'Throttling.AllocationQuota']

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

    def _call_with_prompt(
            self,
            sentence: str,
            label: int,
            _id: int
    ) -> Tuple[bool, Dict[str, Any]]:
        time.sleep(random.randint(1, 20))
        api_key = self.api_keys[_id % len(self.api_keys)]
        response = Generation.call(
            model=self.llm_type,
            prompt=self._process_input_text(sentence, label),
            api_key=api_key,
            stream=True
        )
        response = list(response)[-1]
        if response['status_code'] == HTTPStatus.OK:
            return True, {
                'id': _id,
                'source': sentence,
                's_label': 2 if 'caption' in self.data_file_path.lower() else label,
                'target': response['output']['text'],
                't_label': label if 'caption' in self.data_file_path.lower() else 1 - label,
                # for show
                'api_key': api_key,
                'output_tokens': response['usage']['output_tokens']
            }
        else:
            return False, {
                'code': response['code'],
                'id': _id,
                'label': label,
                'text': sentence
            }

    def _qwen_process(self, _list: List[Dict[str, Any]]):
        success, error, block = [], [], []
        output_tokens_show = {api_key[:7]: 0 for api_key in self.api_keys}
        with ThreadPoolExecutor(max_workers=len(self.api_keys)) as pool:
            label_key = 'label' if 'caption' not in self.data_file_path else 't_label'
            futures = [
                pool.submit(self._call_with_prompt, item['text'], item[label_key], item['id'])
                for item in _list
            ]
            pbar = tqdm(total=len(futures), dynamic_ncols=True)
            for future in as_completed(futures):
                status, result = future.result()
                if status:
                    key = result.pop('api_key')[:7]
                    output_tokens_show[key] += result.pop('output_tokens')
                    success.append(result)
                else:
                    if result['code'] in self.ERROR_CODE:
                        error.append(result)
                    else:
                        block.append(result)
                pbar.update(1)
                pbar.set_postfix(output_tokens_show)
        return success, error, block

    def rewrite(self):
        success_list, error_list, block_list = [], [], []
        success_output_file_path = build_path(self.output_dir, f'{self.llm_type}.csv')
        error_output_file_path = build_path(self.output_dir, f'{self.llm_type}-error.csv')
        block_output_file_path = build_path(self.output_dir, f'{self.llm_type}-block.csv')
        limit_cnt, last_left = 3, 0

        print('load dataset from {}'.format(self.data_file_path))
        data_df = pd.read_csv(self.data_file_path)
        print(f'data size = {data_df.shape[0]}')
        wait_to_process = data_df.to_dict(orient='records')

        start_time = time.time_ns()

        while True:
            success, error, block = self._qwen_process(wait_to_process)

            success_list.extend(success)
            block_list.extend(block)  # do not rewrite again
            wait_to_process = error

            print('success = {}, error = {}, block = {}'.format(len(success), len(error), len(block)))

            wait_to_process_cnt = len(wait_to_process)

            if wait_to_process_cnt != last_left:
                limit_cnt = 3
            else:
                limit_cnt -= 1
            time.sleep(30)
            last_left = wait_to_process_cnt
            if limit_cnt == 0:
                error_list.extend(wait_to_process)
                break

            if wait_to_process_cnt == 0:
                break

        if len(success_list) != 0:
            pd.DataFrame(success_list).sort_values(
                by='id', ascending=True
            ).to_csv(success_output_file_path, index=False)
            print(f'{success_output_file_path} saved !')

        if len(error_list) != 0:
            pd.DataFrame(error_list).sort_values(
                by='id', ascending=True
            ).to_csv(error_output_file_path, index=False)
            print(f'{error_output_file_path} saved !')

        if len(block_list) != 0:
            pd.DataFrame(block_list).sort_values(
                by='id', ascending=True
            ).to_csv(block_output_file_path, index=False)
            print(f'{block_output_file_path} saved !')

        end_time = time.time_ns()
        time_cost = (end_time - start_time) / 1e9
        print(f'Time: {time_cost} s')
