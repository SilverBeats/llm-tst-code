from typing import List, Union

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import build_path, convert_label_to_style
from .base import BaseRewriter


class MistralRewriter(BaseRewriter):
    def __init__(
            self,
            output_dir: str,
            llm_type: str,
            model_dir: str,
            tst_type: str,
            data_file_path: str,
            template: str,
            batch_size: int,
            device: Union[torch.device, str] = 'auto',
            **kwargs
    ):
        super().__init__()
        self.output_dir = output_dir
        self.data_file_path = data_file_path

        self.template = template
        self.tst_type = tst_type
        self.llm_type = llm_type

        self.batch_size = batch_size
        self.device = device

        print(f'load model from {model_dir}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        setattr(self.model.config, 'pad_token', self.tokenizer.pad_token)
        setattr(self.model.config, 'pad_token_id', self.tokenizer.pad_token_id)

    def _process_input_text(self, raw_text: str, label: int):
        if 'caption' in self.data_file_path.lower():
            source_style = 'factual'
            target_style = convert_label_to_style(label, self.tst_type)
        else:
            source_style = convert_label_to_style(label, self.tst_type)
            target_style = convert_label_to_style(1 - label, self.tst_type)
        return self.template.format(source_style, target_style, raw_text)

    def rewrite(self):
        print('load dataset from {}'.format(self.data_file_path))
        data_df = pd.read_csv(self.data_file_path)
        print(f'data size = {data_df.shape[0]}')

        processed = []

        for i in tqdm(range(0, data_df.shape[0], self.batch_size)):
            batch_df = data_df[i: i + self.batch_size]
            batch_text = batch_df['text'].tolist()
            label_key = 'label' if 'caption' not in self.data_file_path else 't_label'
            batch_label = batch_df[label_key].tolist()
            batch_id = batch_df['id'].tolist()

            input_text = [
                self._process_input_text(raw, label)
                for raw, label in zip(batch_text, batch_label)
            ]
            model_inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                padding=True,
                return_attention_mask=False
            ).to(self.device)
            generated_ids = self.model.generate(
                model_inputs['input_ids'],
                max_new_tokens=1000,
                do_sample=True
            )
            sentences: List[str] = self.tokenizer.batch_decode(
                sequences=generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            for _id, s, s_l, t in zip(batch_id, batch_text, batch_label, sentences):
                processed.append({
                    'id': _id,
                    'source': s,
                    's_label': 2 if 'caption' in self.data_file_path.lower() else s_l,
                    'target': t,
                    't_label': s_l if 'caption' in self.data_file_path.lower() else 1 - s_l
                })

        output_file_path = build_path(self.output_dir, f'{self.llm_type}.csv')
        pd.DataFrame(processed).sort_values(by='id').to_csv(output_file_path, index=False)
        print(f'saved at {output_file_path}')
