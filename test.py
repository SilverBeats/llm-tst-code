import pandas as pd
from utils import build_path
from global_config import ACCEPTABLE_LLM, ACCEPTABLE_DATASET, DATASET_TO_TST_TYPE
from collections import defaultdict
from pprint import pp
_dict = defaultdict(dict)

for model in ACCEPTABLE_LLM:
    for dataset in ACCEPTABLE_DATASET:
        total = pd.read_csv(build_path(
            'data', DATASET_TO_TST_TYPE[dataset],
            dataset, 'test.csv'
        )).shape[0]

        fact_size = pd.read_csv(build_path(
            'output', 'special', '0',
            DATASET_TO_TST_TYPE[dataset], dataset,
            'process', f'{model.type}-processed.csv'
        )).shape[0]
        _dict[model.type][dataset] = (total - fact_size) / total * 100
pp(_dict)
