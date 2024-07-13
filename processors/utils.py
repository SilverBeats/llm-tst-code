import re

from utils import norm, remove_emoji


def interval_abbr(s: str, mode: int = 1):
    maps1 = {
        ",": " ,",
        ".": " .",
        "?": " ?",
        "!": " !",
        ":": " :",
        ";": " ;",
        '"': ' " ',
    }
    if mode == 1:
        maps2 = {
            "n't": " n't",
            "'ll": " 'll",
            "'s": " 's",
            "'m": " 'm",
            "s'": "s '",
            "'ve": " 've",
            "'re": " 're",
        }
    elif mode == 2:
        maps2 = {
            "n't": "n t",
            "'ll": " ll",
            "'s": " s",
            "'m": " m",
            "s'": "s ",
            "'ve": " ve",
            "'re": " re",
        }
    else:
        raise NotImplementedError
    for k, v in maps1.items():
        s = s.replace(k, v)
    for k, v in maps2.items():
        s = s.replace(k, v)

    return norm(s)


def interval_punctuation(raw_text: str):
    for p in ['.', '!', '?', '，', '。', '？', ':', '：']:
        if p in raw_text:
            raw_text = raw_text.replace(p, f' {p}')
    return raw_text


def process_mistral(raw_text: str):
    try:
        if re.compile('[\u4e00-\u9fa5]+').search(raw_text):
            return -1
        # delete emoji if exist
        text = remove_emoji(raw_text)
        text = text[text.rfind('[/INST]') + 7:]

        if any(t in raw_text.lower() for t in ["i'm sorry"]):
            return -1

        tags = ['revised sentence:', 'sentence:', 'revised:', 'sentence (revised):', 'sentence revised:']
        for tag in tags:
            i = text.lower().find(tag)
            if i != -1:
                text = text[i + len(tag):]
        text = text.strip()
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        if len(text) == 0:
            return -1
        return norm(interval_punctuation(text))
    except Exception as e:
        return -1


def process_qwen_7b_chat(raw_text: str):
    try:
        if re.compile('[\u4e00-\u9fa5]+').search(raw_text):
            return -1
        # delete emoji if exist
        text = remove_emoji(raw_text)

        if any(t in raw_text.lower() for t in ["i'm sorry"]):
            return -1

        tags = ['sentence:', 'sentence revised:', 'would be:', 'is:']
        for t in tags:
            i = text.lower().find(t)
            if i != -1:
                text = text[i + len(t):]
        # try to find to quote
        text = text.strip()
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        if len(text) == 0:
            return -1

        return norm(interval_punctuation(text))
    except Exception as e:
        return -1


def process_qwen_14b_chat(raw_text: str):
    try:
        if re.compile('[\u4e00-\u9fa5]+').search(raw_text):
            return -1
        # delete emoji if exist
        text = remove_emoji(raw_text)

        if any(t in raw_text.lower() for t in ["i'm sorry"]):
            return -1

        tags = ['sentence:', 'style:', 'as follows:', 'rewritten:', 'meaning:', 'revised sentence:']
        for t in tags:
            i = text.lower().find(t)
            if i != -1:
                text = text[i + len(t):]
        # try to find to quote
        text = text.strip()
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        if len(text) == 0:
            return -1

        return norm(interval_punctuation(text))
    except Exception as e:
        return -1


def process_7b_llama2(raw_text: str):
    i = raw_text.find('[/INST]')
    text = remove_emoji(raw_text[i + 1:])
    r = list(filter(lambda s: s.strip() != '', text.split('\n')))
    if any(item in r[0].lower() for item in ['sorry', 'apologize']):
        return -1
    if len(r) != 2:
        return -1
    gen_text = r[1].strip()
    if gen_text[0] == gen_text[-1] == '"':
        gen_text = gen_text[1:-1]
    return norm(interval_punctuation(gen_text))


def process_13b_llama2(raw_text: str):
    i = raw_text.find('[/INST]')
    text = remove_emoji(raw_text[i + 1:])
    r = list(filter(lambda s: s.strip() != '', text.split('\n')))
    if any(item in r[0].lower() for item in ['sorry', 'apologize']):
        return -1
    if len(r) != 2:
        return -1
    gen_text = r[1].strip()
    if gen_text[0] == gen_text[-1] == '"':
        gen_text = gen_text[1:-1]
    return norm(interval_punctuation(gen_text))


def process_gpt(raw_text: str):
    try:
        # delete emoji if exist
        text = remove_emoji(raw_text)
        tags = ['revised sentence:', 'revised:']
        for tag in tags:
            i = text.lower().find(tag)
            if i != -1:
                text = text[i + len(tag):]

        text = text.strip()
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        if len(text) == 0:
            return -1
        return norm(interval_punctuation(text))
    except Exception as e:
        return -1
