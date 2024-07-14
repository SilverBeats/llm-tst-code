import os
import re
from typing import List

import emoji

from constant import LLMType, AccModelType, FluencyModelType


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


def convert_to_llm_type(llm_type: str) -> LLMType:
    for item in LLMType:
        if item.type == llm_type:
            return item
    raise ValueError


def convert_str_to_acc_model_type(acc_model_type: str) -> AccModelType:
    for item in AccModelType:
        if item.value == acc_model_type:
            return item
    raise ValueError


def convert_str_to_fluency_model_type(flu_model_type: str) -> FluencyModelType:
    for item in FluencyModelType:
        if item.value == flu_model_type:
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


def get_dir_file_path(
        dir_name: str,
        file_ext: List[str] = None,
        skip_dir_names: List[str] = None,
        skip_file_names: List[str] = None,
        is_abs: bool = False
):
    if is_abs:
        dir_name = os.path.abspath(dir_name)
    if file_ext is None:
        file_ext = []
    if skip_dir_names is None:
        skip_dir_names = []
    if skip_file_names is None:
        skip_file_names = []
    # 获得所有的文件夹、文件
    arr = []
    all_file_and_dir_name = os.listdir(dir_name)
    for file_or_dir in all_file_and_dir_name:
        full_path = os.path.join(dir_name, file_or_dir)
        # 如果是目录, 递归
        if os.path.isdir(full_path):
            if file_or_dir in skip_dir_names:
                continue
            arr.extend(get_dir_file_path(full_path, file_ext, skip_dir_names, skip_file_names, is_abs))
        else:  # 如果是文件
            if file_or_dir in skip_file_names:
                continue
            if len(file_ext) > 0:
                if any(full_path.endswith(ext) for ext in file_ext):
                    arr.append(full_path)
            else:
                arr.append(full_path)
    return arr

