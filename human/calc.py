from typing import List
from pprint import pp
from scipy.stats import kendalltau, spearmanr
import os
import pandas as pd
import numpy as np
# 计算的数据集
dataset = 'amazon'
sheet_names = ['acc', 'content', 'fluency']
model_cnts = 15
# 标注结果根目录
file_root_dir = r'E://我的坚果云/文本风格转换/10%/'
# 获取所有标注结果文件
annotation_names = os.listdir(file_root_dir)
# 记录结果
# {acc: {name1: [], name2: [], ...}, content: {}, fluency:{}}
result_record = {k: {} for k in sheet_names}

# 读数据
for name in annotation_names:  # name 标注者姓名
    full_file_path = os.path.join(file_root_dir, name, f'{dataset}.xlsx')
    assert os.path.exists(full_file_path)
    for sheet_name in sheet_names:
        # 读取数据
        df = pd.read_excel(full_file_path, sheet_name=sheet_name)
        r_df = df[df['id'].isna()]  # 提取结果行
        r_df = r_df.dropna(subset=['model0'])  # 过滤未标注的行
        r_dict = (r_df[[f'model{i}'for i in range(model_cnts)]].sum(axis=0) / r_df.shape[0]).to_dict()
        result = sorted([(k, v) for k, v in r_dict.items()], key=lambda item: item[1], reverse=False)
        no_list = [0] * model_cnts
        for no, (model_name, v) in enumerate(result):
            no_list[int(model_name[5:])] = no + 1
        result_record[sheet_name][name] = no_list
print(result_record)

kendalltau_result = {}
spearmanr_result = {}
for sheet_name in sheet_names:
    k_taus, k_p_values = [], []
    s_correlation, s_p_values = [], []
    values: List[List[int]] = list(result_record[sheet_name].values())
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        t, p1 = kendalltau(values[i], values[j])
        c, p2 = spearmanr(values[i], values[j])
        k_taus.append(t)
        k_p_values.append(p1)
        s_correlation.append(c)
        s_p_values.append(p2)
    kendalltau_result[sheet_name] = {
        'mean_tau': np.asarray(k_taus).mean(),
        'mean_p': np.asarray(k_p_values).mean(),
    }
    spearmanr_result[sheet_name] = {
        'mean_correlation': np.asarray(s_correlation).mean(),
        'mean_p': np.asarray(s_p_values).mean(),
    }
pp(kendalltau_result)
pp(spearmanr_result)