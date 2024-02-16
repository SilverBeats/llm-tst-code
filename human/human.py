import json
import os

import pandas as pd
from transformers import set_seed

SEED = 43
DATASETS = ['imagecaption', 'gender', 'political']
OUTPUT_DIR = 'output'
SHEET_NAMES = ['acc', 'content', 'fluency']
EACH_STYLE_SAMPLES = [50, 50]


def func(s: int, dataset: str):
    if s is None:
        return s
    if pd.isna(s):
        return s
    if dataset in ['yelp', 'amazon']:
        return '积极' if s == 1 else '消极'

    if dataset == 'political':
        return '共和党' if s == 0 else '民主党'

    if dataset == 'gender':
        return '男' if s == 0 else '女'

    if dataset == 'imagecaption':
        if s == 0:
            return '幽默'
        elif s == 2:
            return '事实'
        else:
            return '浪漫'

    return 'unknown'


def process(dataset: str):
    file_names = os.listdir(dataset)
    model_order = {}
    has_necessary_column = False
    df = []
    for i, f_n in enumerate(file_names):
        model_order[f'model{i}'] = f_n[:-4]
        cur_df = pd.read_csv(os.path.join(dataset, f_n))
        cur_df.rename(columns={'target': f'model{i}'}, inplace=True)
        if not has_necessary_column and 'source' in cur_df.columns:
            df.append(cur_df[['source', 's_label', 't_label', f'model{i}']])
            has_necessary_column = True
        else:
            df.append(cur_df[[f'model{i}']])
    # 所有数据
    df = pd.concat(df, axis=1)

    # 获取所有的标签
    labels = df['t_label'].unique().tolist()

    # 抽取每个标签的样本
    sample_df = pd.concat([
        df[df['t_label'] == label].sample(frac=1).sample(n)
        for label, n in zip(labels, EACH_STYLE_SAMPLES)
    ])
    # 每一行下面添加空行
    new_df = pd.DataFrame()
    for i, row in sample_df.iterrows():
        new_df = new_df.append(row)
        new_df = new_df.append(pd.Series([], dtype='float64'), ignore_index=True)

    new_df['id'] = sum([[(row_num // 2) + 1, ''] for row_num in range(0, new_df.shape[0], 2)], [])
    # 列排序
    new_order = ['id', 'source', 's_label', 't_label'] + [f'model{i}' for i in range(len(file_names))]
    new_df = new_df[new_order]
    # rename columns
    new_df['s_label'] = new_df['s_label'].apply(func, args=(dataset,))
    new_df['t_label'] = new_df['t_label'].apply(func, args=(dataset,))
    new_df.rename(columns={
        'id': 'id',
        'source': '源句',
        's_label': '源风格',
        't_label': '目标风格'
    }, inplace=True)

    with open(os.path.join(OUTPUT_DIR, f'{dataset}_order.json'), 'w', encoding='utf-8') as f:
        json.dump(model_order, f, ensure_ascii=False, indent=4)

    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, f'{dataset}.xlsx')) as writer:
        for sheet_name in SHEET_NAMES:
            new_df.to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    set_seed(43)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dataset in DATASETS:
        process(dataset)
        print(f'process {dataset} is success .')


if __name__ == '__main__':
    main()
