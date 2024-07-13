from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_score import score
from fasttext.FastText import _FastText
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import GPT2Tokenizer


@torch.no_grad()
def calc_acc_by_plm(texts: List[str], ground_labels: List[int], model, tokenizer, batch_size, device):
    assert len(texts) == len(ground_labels)
    size = len(ground_labels)
    right_cnt = 0

    for i in tqdm(range(0, size, batch_size), dynamic_ncols=True, desc='calculate acc ...'):
        batch_text = texts[i: i + batch_size]
        labels = torch.LongTensor(ground_labels[i: i + batch_size])
        inputs = tokenizer(
            text=batch_text,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
            padding=True,
        ).to(device)
        logits = model(**inputs).logits
        pred_labels = logits.argmax(dim=-1).detach().cpu()
        right_cnt += pred_labels.eq(labels).sum().item()

    return 100 * right_cnt / size


@torch.no_grad()
def eval_model_loss(model: nn.Module, tokenizer: GPT2Tokenizer, text_list: List[str], device: str, batch_size: int):
    model.eval()

    total_loss = 0
    total_sample = 0
    for i in tqdm(range(0, len(text_list), batch_size), dynamic_ncols=True):
        batch_text = text_list[i: i + batch_size]
        input_ids = tokenizer(batch_text, return_tensors='pt', padding=True)['input_ids'].to(device)

        # (bs, seq_len - 1), [1:]
        lm_labels = input_ids.clone().detach()[:, 1:]
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100

        lm_logits = model(input_ids).logits  # (bs, seq_len, vocab_size)
        lm_logits = lm_logits[:, :-1]

        lm_loss = F.cross_entropy(
            input=lm_logits.reshape(-1, lm_logits.shape[-1]),
            target=lm_labels.reshape(-1),
            ignore_index=-100,
            reduction='sum'
        )
        total_loss += lm_loss.item()
        total_sample += (lm_labels != -100).sum().item()

    mean_loss = total_loss / total_sample
    mean_ppl = np.exp(mean_loss)
    print(f'\ncur_loss={mean_loss}, cur_ppl={mean_ppl}')
    return mean_ppl


def eval_acc_by_fasttext(texts: List[str], ground_labels: List[int], model: _FastText):
    predict_labels = [int(item[0][-1]) for item in model.predict(texts)[0]]
    return_dict = {
        "f1": f1_score(ground_labels, predict_labels),
        "acc": accuracy_score(ground_labels, predict_labels),
        "p@1": precision_score(ground_labels, predict_labels),
        "r@1": recall_score(ground_labels, predict_labels)
    }
    return return_dict


def calc_corpus_bleu(ref_list: List[str], hyp_list: List[str], tokenizer: Callable = str.split):
    hyp_texts = [tokenizer(sent) for sent in hyp_list]
    ref_texts = [[tokenizer(sent)] for sent in ref_list]
    return corpus_bleu(ref_texts, hyp_texts) * 100


def calc_sentence_bleu(ref_list: List[str], hyp_list: List[str], tokenizer: Callable):
    bleu = 0
    hyp_texts = [tokenizer(sent) for sent in hyp_list]
    ref_texts = [[tokenizer(sent)] for sent in ref_list]
    for ref, hyp in zip(ref_texts, hyp_texts):
        bleu += sentence_bleu(ref, hyp)
    return bleu / len(ref_list) * 100.0


def calc_bert_score(ref_list: List[str], hyp_list: List[str], bert_score_config: dict):
    P, R, b_F = score(
        cands=hyp_list,
        refs=ref_list,
        **bert_score_config
    )
    return {
        'bert_score_P ': round(P.mean().item(), 6),
        'bert_score_R': round(R.mean().item(), 6),
        'bert_score_F': round(b_F.mean().item(), 6),
    }
