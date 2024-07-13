import json
import math
import os.path
from pprint import pp
from typing import Union

import fasttext
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)

from evaluators.utils import eval_acc_by_fasttext
from utils import build_path, rm_dir
from .dataset import get_dataloader


def train_fasttext_classifier(
        train_file_path: str,
        dev_file_path: str,
        ckpt_file_path: str,
        output_dir: str,
        train_args: dict
):
    old_acc, old_f1 = 0, 0
    if os.path.exists(build_path(output_dir, 'eval.json')):
        with open(build_path(output_dir, 'eval.json'), 'r', encoding='utf-8') as f:
            d = json.load(f)
            old_acc = d['acc']
            old_f1 = d['f1']
    print(f'exists best acc = {old_acc}, best f1 = {old_f1}')

    model = fasttext.train_supervised(**train_args, input=train_file_path)
    print(f'model test on {dev_file_path}')
    dev_df = pd.read_csv(dev_file_path)
    save_dict = eval_acc_by_fasttext(dev_df['text'].apply(str.lower).tolist(), dev_df['label'].tolist(), model)
    pp(save_dict)

    if save_dict['acc'] > old_acc or (save_dict['acc'] == old_acc and save_dict['f1'] > old_f1):
        model.save_model(ckpt_file_path)
        print(f'model saved at {ckpt_file_path}')
        with open(build_path(output_dir, 'eval.json'), 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, indent=4, ensure_ascii=False)
        with open(build_path(output_dir, 'train_args.json'), 'w', encoding='utf-8') as f:
            json.dump(train_args, f, indent=4, ensure_ascii=False)


@torch.no_grad()
def _do_classifier_evaluate(model, dataloader) -> float:
    model.eval()
    right_cnt = 0
    total_samples = 0
    for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, desc='evaluating'):
        predict_logits = model(**batch).logits  # (bs, 2)
        predict_label = predict_logits.argmax(1).cpu()  # (bs, )
        right_cnt += (batch['labels'].cpu() == predict_label).sum().item()
        total_samples += predict_label.shape[0]
    model.train()
    acc = right_cnt / total_samples
    return acc * 100


def train_classifier_bert(
        pretrained_model_name_or_dir: str,
        data_dir: str,
        output_dir: str,
        seed: int = 42,
        epochs: int = 5,
        train_batch_size: int = 32,
        dev_batch_size: int = 32,
        test_batch_size: int = 32,
        lr: float = 1e-5,
        max_seq_len: int = 510,
        max_grad_norm: float = 1.0,
        warmup: Union[float, int] = 0.1,
        device: Union[str, torch.device] = 'cuda:0',
        do_test: bool = True,
        **kwargs
):
    set_seed(seed)
    print(f'create model from {pretrained_model_name_or_dir}')
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_dir)
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_dir,
        num_labels=2
    ).train().to(device)

    print(f'load data from {data_dir}')
    train_file_path = build_path(data_dir, 'train.csv')
    dev_file_path = build_path(data_dir, 'dev.csv')
    train_loader = get_dataloader(train_file_path, 'train', max_seq_len, tokenizer, train_batch_size, device)
    dev_loader = get_dataloader(dev_file_path, 'dev', max_seq_len, tokenizer, dev_batch_size, device)

    print('create optimizer and schedule')
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = warmup if not isinstance(warmup, float) else int(warmup * num_training_steps)
    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        no_deprecation_warning=True
    )

    if num_warmup_steps != 0:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_acc = -1
    best_ckpt_dir = ''
    pbar = tqdm(total=num_training_steps, desc='training', dynamic_ncols=True)
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model(**batch).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if num_warmup_steps != 0:
                lr_scheduler.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            pbar.set_postfix({'loss': round(loss.item(), 6)})
            pbar.update(1)
            torch.cuda.empty_cache()
        eval_acc = _do_classifier_evaluate(model, dev_loader)
        print(f'best acc = {best_acc}, current acc = {eval_acc}')
        if best_acc <= eval_acc:
            print('update !')
            print(f'remove the old ckpt path {best_ckpt_dir}')
            rm_dir(best_ckpt_dir)
            best_acc = eval_acc
            best_ckpt_dir = build_path(output_dir, epoch)
            model.save_pretrained(best_ckpt_dir)
            tokenizer.save_pretrained(best_ckpt_dir)
            print(f'new ckpt saved at {best_ckpt_dir}')
    save_dict = {
        'best_acc_on_dev': best_acc,
    }
    if do_test:
        test_file_path = build_path(data_dir, 'test.csv')
        test_loader = get_dataloader(test_file_path, 'test', max_seq_len, tokenizer, test_batch_size, device)
        del model
        model = BertForSequenceClassification.from_pretrained(best_ckpt_dir, num_labels=2).eval().to(device)
        save_dict['acc_on_test'] = _do_classifier_evaluate(model, test_loader)

    with open(build_path(best_ckpt_dir, 'eval.json'), 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=4)


@torch.no_grad()
def _do_fluency_evaluate(model, tokenizer, dataloader):
    model.eval()

    total_sample = 0
    total_loss = 0

    for batch in tqdm(dataloader, total=len(dataloader), desc='evaluating', dynamic_ncols=True):
        if 'labels' in batch:
            batch.pop('labels')
        lm_labels = batch['input_ids'].detach().clone()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100

        logits = model(**batch).logits  # (bs, seq_len, vocab_size)
        lm_predict = logits[:, :-1]  # (bs, seq_len - 1, vocab_size)
        loss = torch.nn.functional.cross_entropy(
            input=lm_predict.reshape(-1, lm_predict.shape[-1]),
            target=lm_labels[:, 1:].reshape(-1),
            reduction='sum',
            ignore_index=-100
        )
        total_loss += loss.item()
        total_sample += (lm_labels != -100).sum()
    model.train()
    mean_loss = total_loss / total_sample
    mean_ppl = math.exp(mean_loss)

    return mean_ppl


def train_fluency_gpt2(
        pretrained_model_name_or_dir: str,
        data_dir: str,
        output_dir: str,
        seed: int = 42,
        epochs: int = 5,
        train_batch_size: int = 32,
        dev_batch_size: int = 32,
        test_batch_size: int = 32,
        lr: float = 1e-5,
        max_seq_len: int = 1024,
        max_grad_norm: float = 1.0,
        warmup: Union[float, int] = 0.1,
        device: Union[str, torch.device] = 'cuda:0',
        do_test: bool = True,
        **kwargs
):
    set_seed(seed)

    print(f'create model from {pretrained_model_name_or_dir}')
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_dir).train().to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_dir)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    print(f'load data from {data_dir}')
    train_file_path = build_path(data_dir, 'train.csv')
    dev_file_path = build_path(data_dir, 'dev.csv')
    train_loader = get_dataloader(train_file_path, 'train', max_seq_len, tokenizer, train_batch_size, device)
    dev_loader = get_dataloader(dev_file_path, 'dev', max_seq_len, tokenizer, dev_batch_size, device)

    print('create optimizer and schedule')
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = warmup if not isinstance(warmup, float) else int(warmup * num_training_steps)
    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        no_deprecation_warning=True
    )
    if num_warmup_steps != 0:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_ppl = 1e30
    best_ckpt_dir = ''

    pbar = tqdm(total=num_training_steps, desc='training', dynamic_ncols=True)
    for epoch in range(epochs):
        for batch in train_loader:
            if 'labels' in batch:
                batch.pop('labels')
            lm_labels = batch['input_ids'].detach().clone()
            lm_labels[lm_labels == tokenizer.pad_token_id] = -100
            loss = model(**batch, labels=lm_labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if num_warmup_steps != 0:
                lr_scheduler.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            pbar.set_postfix({'loss': round(loss.item(), 6)})
            pbar.update(1)
            torch.cuda.empty_cache()
        eval_ppl = _do_fluency_evaluate(model, tokenizer, dev_loader)
        print(f'best ppl = {best_ppl}, current ppl = {eval_ppl}')
        if best_ppl > eval_ppl:
            print('update !')
            print(f'remove the old ckpt path: {best_ckpt_dir}')
            rm_dir(best_ckpt_dir)
            best_ppl = eval_ppl
            best_ckpt_dir = build_path(output_dir, epoch)
            model.save_pretrained(best_ckpt_dir)
            tokenizer.save_pretrained(best_ckpt_dir)
            print(f'new ckpt saved at {best_ckpt_dir}')

    save_dict = {
        'best_ppl_on_dev': best_ppl,
    }
    if do_test:
        test_file_path = build_path(data_dir, 'test.csv')
        test_loader = get_dataloader(test_file_path, 'test', max_seq_len, tokenizer, test_batch_size, device)
        del model
        model = GPT2LMHeadModel.from_pretrained(best_ckpt_dir).eval().to(device)
        save_dict['ppl_on_test'] = _do_fluency_evaluate(model, tokenizer, test_loader)

    with open(build_path(best_ckpt_dir, 'eval.json'), 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=4)
