import os.path
from typing import Callable, Dict, Union, List

import torch

from constant import DATASET_TO_TST_TYPE, LLMType, Task
from utils import build_path, process_template
from .llama2 import Llama2Rewriter
from .mistral import MistralRewriter
from .openai import OpenAIRewriter
from .qwen import QwenRewriter

MAP_ABBR_TO_CLS: Dict[str, Callable] = {
    # gpt
    LLMType.gpt_3_5_turbo.abbr: OpenAIRewriter,
    # qwen
    LLMType.qwen_7b_chat.abbr: QwenRewriter,
    # mistral
    LLMType.mistral_7b_instruct.abbr: MistralRewriter,
    # llama2
    LLMType.llama2_7b_chat_hf.abbr: Llama2Rewriter,
}


class Rewriter:

    def __init__(
            self,
            dataset: str,
            llm_type: LLMType,
            output_dir: str = 'output',
            split: str = 'test',
            data_dir: str = 'data',
            device: Union[str, torch.device] = 'cuda:0',
            batch_size: int = 8,
            llm_model_dir: str = None,
            template_type: str = 'common',
            template_idx: int = 0,
            api_keys: Dict[str, List[str]] = None,
            templates: Dict[str, Union[List[str], Dict[str, List[str]]]] = None,
            **kwargs
    ):
        tst_type = DATASET_TO_TST_TYPE[dataset]
        data_dir = build_path(data_dir, tst_type, dataset)

        data_file_path = build_path(data_dir, f'{split}.csv')
        assert os.path.exists(data_file_path)

        output_dir = build_path(
            output_dir, template_type, template_idx,
            tst_type, dataset, str(Task.R)
        )
        os.makedirs(output_dir, exist_ok=True)

        rewriter_kwargs = {
            # common parameter
            'output_dir': output_dir,
            'llm_type': llm_type.type,
            'tst_type': tst_type,
            'data_file_path': data_file_path,
            'template': process_template(
                llm_type,
                templates[template_type][dataset][template_idx]
                if template_type == 'special'
                else templates[template_type][template_idx]
            ),
            # parameters used when need to load the model
            'batch_size': batch_size,
            'device': device,
            'model_dir': llm_model_dir,
            # parameters when need to use API
            'api_keys': api_keys.get(llm_type.abbr, [])
        }

        rewriter_cls = MAP_ABBR_TO_CLS.get(llm_type.abbr, None)
        if rewriter_cls is None:
            raise NotImplementedError

        self.rewriter = rewriter_cls(**rewriter_kwargs)

    def rewrite(self):
        self.rewriter.rewrite()
