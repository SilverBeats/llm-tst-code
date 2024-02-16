import os
import re

import emoji

from global_config import LLMType


def remove_emoji(s):
    s = emoji.demojize(s)
    # s = re.sub(emoji.get_emoji_regexp(), "", s)
    s = re.sub(r':\S+?:', ' ', s)
    return s


def norm(s: str) -> str:
    arr = []
    for item in s.split():
        arr.append(item.strip())
    return ' '.join(arr)


def build_path(*args):
    args = tuple(str(item) for item in args)
    return os.path.join(*args)


def rm_file(path: str):
    if os.path.exists(path) and os.path.isfile(path):
        os.remove(path)


def rm_dir(path: str):
    if os.path.exists(path) and os.path.isdir(path):
        for f_name in os.listdir(path):
            sub_path = build_path(path, f_name)
            if os.path.isdir(sub_path):
                rm_dir(sub_path)
            else:
                rm_file(sub_path)
        os.rmdir(path)


def convert_label_to_style(label: int, tst_type: str) -> str:
    if tst_type == 'pos-neg':
        return 'negative' if label == 0 else 'positive'
    elif tst_type == 'male-female':
        return 'male' if label == 0 else 'female'
    elif tst_type == 'paper-news':
        return 'paper' if label == 0 else 'news'
    elif tst_type == 'republican-democratic':
        return 'republican' if label == 0 else 'democratic'
    elif tst_type == 'romantic-humorous':
        return 'humorous' if label == 0 else 'romantic'
    else:
        raise NotImplementedError(f'Not implemented tst_type: {tst_type}')


def get_LLMType(llm_type: str) -> LLMType:
    for item in LLMType:
        if item.type == llm_type:
            return item
    raise ValueError


def process_template(model: LLMType, template: str):
    pos = template.find('Sentence: {}')
    assert pos != -1

    if model.abbr.startswith('mistral'):
        template = f'<s> [INST] {template} [/INST]'
    elif model.abbr.startswith('llama2'):
        template = f"<s> [INST] <<SYS>> {template[:pos]} <</SYS>> {template[pos:]} [/INST]"
    elif any(model.abbr.startswith(k) for k in ['gpt', 'qwen']):
        template = template
    return norm(template)
