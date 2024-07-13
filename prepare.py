"""
The main function of this document is to prepare a model for evaluating the results.
"""
import os
from argparse import ArgumentParser
from pprint import pp

from global_config import ACCEPTABLE_DATASET, DATASET_TO_TST_TYPE
from prepares import *
from utils import build_path

DATASET_TO_FUN = {
    'yelp': {
        'classifier': train_yelp_classifier,
        'fluency': train_yelp_d_ppl
    },
    'amazon': {
        'classifier': train_amazon_classifier,
        'fluency': train_amazon_d_ppl
    },
    'imagecaption': {
        'classifier': train_caption_classifier,
        'fluency': train_caption_d_ppl
    },
    'gender': {
        'classifier': train_gender_classifier,
        'fluency': train_gender_d_ppl
    },
    'political': {
        'classifier': train_political_classifier,
        'fluency': train_political_d_ppl
    }
}


def get_args():
    parser = ArgumentParser()
    # Revise frequently
    parser.add_argument('--dataset', type=str, choices=ACCEPTABLE_DATASET, required=True)
    parser.add_argument('--task', type=str, choices=['classifier', 'fluency'], required=True)
    parser.add_argument('--pretrained_dir_or_name', type=str, default='')

    # Basically not modify
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='output')
    return vars(parser.parse_args())


def main():
    config = get_args()

    config['data_dir'] = build_path(
        config['data_dir'],
        DATASET_TO_TST_TYPE[config['dataset']],
        config['dataset']
    )
    config['output_dir'] = build_path(
        config['output_dir'],
        config['task'],
        DATASET_TO_TST_TYPE[config['dataset']],
        config['dataset']
    )
    os.makedirs(config['output_dir'], exist_ok=True)
    pp(config)

    if config['task'] not in DATASET_TO_FUN[config['dataset']].keys():
        raise NotImplementedError(
            'Data set {} does not support task {}'
            .format(config['dataset'], config['task'])
        )
    task = config['task']
    print('Perform Task {}'.format(task))
    DATASET_TO_FUN.get(config['dataset']).get(task)(**config)


if __name__ == '__main__':
    main()
