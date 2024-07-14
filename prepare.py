import os
from pprint import pp

from omegaconf import DictConfig, OmegaConf
from constant import ACCEPTABLE_DATASET, DATASET_TO_TST_TYPE
from prepares import *
from utils import build_path, convert_str_to_fluency_model_type, convert_str_to_acc_model_type

TASK_TO_FUN = {
    'classifier': train_base_classifier,
    'fluency': train_base_fluency
}


def main(config: DictConfig):
    assert config['dataset'] in ACCEPTABLE_DATASET
    assert config['task'] in TASK_TO_FUN.keys()
    if config['task'] == 'classifier':
        config['model_type'] = convert_str_to_acc_model_type(config['model_type'])
    elif config['task'] == 'fluency':
        config['model_type'] = convert_str_to_fluency_model_type(config['model_type'])

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

    task = config['task']
    print('Perform task = {}, dataset = {}'.format(task, config['dataset']))
    TASK_TO_FUN.get(task)(
        output_dir=config['output_dir'],
        data_dir=config['data_dir'],
        device=config['device'],
        seed=config['seed'],
        model_type=config['model_type'],
        train_config=config['train_config'][config['task']][config['model_type']]
    )


if __name__ == '__main__':
    config_file_path = 'config/prepare.yaml'
    main(OmegaConf.load(config_file_path))
