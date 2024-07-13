import json
import os
from argparse import ArgumentParser
from pprint import pp

import torch

from evaluators.amazon import AmazonEvaluator
from evaluators.caption import CaptionEvaluator
from evaluators.gender import GenderEvaluator
from evaluators.political import PoliticalEvaluator
from evaluators.yelp import YelpEvaluator
from utils import build_path
from global_config import DATASET_TO_TST_TYPE

DATASET_TO_EVALUATOR = {
    'yelp': YelpEvaluator,
    'amazon': AmazonEvaluator,
    'imagecaption': CaptionEvaluator,
    'gender': GenderEvaluator,
    'political': PoliticalEvaluator,
}


def get_args():
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, required=True)
    args.add_argument('--acc_model_dir_or_path', type=str, required=True)
    args.add_argument('--d_ppl_model_dir', type=str, required=True)

    # infrequent revisions
    args.add_argument('--baseline_dir', type=str, default='baselines')
    args.add_argument('--skip_models', type=str, default=None, nargs='+')
    args.add_argument('--data_dir', type=str, default='baselines')
    args.add_argument('--device', type=str, default='cuda:0')
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--g_ppl_model_dir', type=str, default='pretrained_models/gpt2-large')
    args.add_argument('--bert_score_model_dir', type=str, default='pretrained_models/roberta-large')
    args.add_argument('--bert_score_model_layers', type=int, default=17)
    return vars(args.parse_args())


def main():
    config = get_args()

    config['data_dir'] = build_path(config['baseline_dir'], config['dataset'])

    # get all models
    file_names = list(filter(lambda f_n: f_n.endswith('csv'), os.listdir(config['data_dir'])))
    if config['skip_models'] is not None:
        file_names = [f for f in file_names if f[:-4] not in config['skip_models']]

    config['ref_file_path'] = None
    if config['dataset'] in ['imagecaption', 'amazon', 'yelp']:
        config['ref_file_path'] = build_path(
            'data',  DATASET_TO_TST_TYPE[config['dataset']],
            config['dataset'], 'reference.csv'
        )
    config['models'] = [f[:-4] for f in file_names]
    pp(config)

    eval_result = {}
    wrong_models = []
    for model in config['models']:
        config['data_file_path'] = build_path(config['data_dir'], f'{model}.csv')
        try:
            evaluator = DATASET_TO_EVALUATOR[config['dataset']](**config)
            eval_result[model] = evaluator.evaluate()
        except Exception as e:
            print(e)
            wrong_models.append(model)
    if len(wrong_models) > 0:
        print(f'models = {wrong_models} goes wrong .')
    pp(eval_result)
    output_file_path = build_path(config['data_dir'], 'eval.json')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, indent=4, ensure_ascii=False)
    print(f'saved at {output_file_path}')


if __name__ == '__main__':
    main()
