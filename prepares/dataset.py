from typing import List, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


class TSTDataset(Dataset):
    def __init__(self, file_path: str):
        super().__init__()
        assert file_path.endswith('.csv')
        df = pd.read_csv(file_path)
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.size = df.shape[0]

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return self.size


def collate_fn(
        batch: List[Tuple[str, int]],
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        device: Union[str, torch.device]
):
    batch_text = [item[0].lower() for item in batch]
    batch_label = torch.LongTensor([item[1] for item in batch]).to(device)
    inputs = dict(tokenizer(
        text=batch_text,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt'
    ).to(device))

    return {
        'labels': batch_label,
        **inputs,  # input_ids, attention_mask, token_type_ids
    }


def get_dataloader(
        data_file_path: str,
        split: str,
        max_len: int,
        tokenizer: PreTrainedTokenizer,
        batch_size,
        device: Union[torch.device, str],
) -> DataLoader:
    return DataLoader(
        dataset=TSTDataset(data_file_path),
        batch_size=batch_size,
        shuffle=True if split == 'train' else False,
        collate_fn=lambda x: collate_fn(x, tokenizer=tokenizer, max_len=max_len, device=device)
    )
