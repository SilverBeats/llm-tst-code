import json
import os
from typing import List

from omegaconf import DictConfig, OmegaConf

from constant import DATASET_TO_TST_TYPE, ACCEPTABLE_DATASET
from evaluators import Evaluator
from utils import build_path, get_dir_file_path, convert_str_to_acc_model_type, convert_str_to_fluency_model_type


def check_config(config: DictConfig):
    assert config['dataset'] in ACCEPTABLE_DATASET
    if config['do_eval']:
        target_eval_config = config['eval_config'][config['dataset']]
        target_eval_config['acc_model_type'] = convert_str_to_acc_model_type(target_eval_config['acc_model_type'])
        target_eval_config['fluency_model_type'] = convert_str_to_fluency_model_type(
            target_eval_config['fluency_model_type'])


if __name__ == '__main__':
    baseline_config_path = 'config/baseline.yaml'
    eval_config_path = 'config/main.yaml'

    baseline_config = OmegaConf.load(baseline_config_path)
    dataset = baseline_config['dataset']
    eval_config = OmegaConf.load(eval_config_path)['eval_config']

    # get all baselines files
    file_paths: List[str] = get_dir_file_path(
        dir_name=baseline_config['baseline_dir'],
        file_ext=['.csv'],
        skip_file_names=[f'{s_f_n}.csv' for s_f_n in baseline_config['skip_models']]
        if baseline_config['skip_models'] else None
    )

    ref_file_path = build_path(
        baseline_config['ref_data_dir'],
        DATASET_TO_TST_TYPE[baseline_config['dataset']],
        baseline_config['dataset'],
        'reference.csv'
    )
    if not os.path.exists(ref_file_path):
        ref_file_path = None

    all_eval_result, wrong_models = {}, []
    for f_p in file_paths:
        model_name = os.path.splitext(os.path.basename(f_p))[0]
        try:
            cur_model_eval_result = Evaluator(
                dataset=baseline_config['dataset'],
                data_file_path=f_p,
                ref_file_path=ref_file_path,
                **eval_config['same'],
                **eval_config[baseline_config['dataset']]
            ).evaluate()
            all_eval_result[model_name] = cur_model_eval_result
        except Exception as e:
            print(e)
            wrong_models.append(model_name)
    if len(wrong_models) > 0:
        print(f'models = {wrong_models} goes wrong .')
    output_file_path = build_path(os.path.dirname(file_paths[0]), 'eval.json')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_eval_result, f, ensure_ascii=False, indent=4)
    print('saved at {}'.format(output_file_path))
