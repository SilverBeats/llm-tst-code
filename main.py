import json
import os
import warnings
from pprint import pp

from omegaconf import DictConfig, OmegaConf

from constant import ACCEPTABLE_DATASET, DATASET_TO_TST_TYPE, Task
from evaluators import Evaluator
from processors import Processor
from rewriters import Rewriter
from utils import build_path, convert_to_llm_type, convert_str_to_fluency_model_type, convert_str_to_acc_model_type


def check_config(config: DictConfig):
    assert config['dataset'] in ACCEPTABLE_DATASET
    config['llm_type'] = convert_to_llm_type(config['llm_type'])

    if config['do_eval']:
        target_eval_config = config['eval_config'][config['dataset']]
        target_eval_config['acc_model_type'] = convert_str_to_acc_model_type(target_eval_config['acc_model_type'])
        target_eval_config['fluency_model_type'] = convert_str_to_fluency_model_type(
            target_eval_config['fluency_model_type'])


def main(config: DictConfig):
    check_config(config)
    pp(config)
    llm_type = config['llm_type']

    do_rewrite, do_process, do_eval = config.pop('do_rewrite'), config.pop('do_process'), config.pop('do_eval')
    rewrite_config, eval_config = config.pop('rewrite_config'), config.pop('eval_config')

    if do_rewrite:
        Rewriter(**config, **rewrite_config).rewrite()
        print('{} rewriting dataset {} has been completed.'.format(str(llm_type), config['dataset']))

    if do_process:
        Processor(**config).process()
        print('The rewrite of the processing {} on dataset {} is complete.'.format(str(llm_type), config['dataset']))

    if do_eval:
        tst_type = DATASET_TO_TST_TYPE[config['dataset']]
        processed_dir = build_path(
            config['output_dir'], config['template_type'], config['template_idx'],
            tst_type, config['dataset'], str(Task.P)
        )
        data_file_path = build_path(processed_dir, f'{str(llm_type)}-processed.csv')
        human_process_file_path = build_path(processed_dir, f'{str(llm_type)}-processed-human.csv')
        if os.path.exists(human_process_file_path):
            warnings.warn('!!!!!!!!! Program pause !!!!!!!!!')
            warnings.warn(
                f"""
                A data file requiring manual processing has been detected.
                Manually repair the data in the `target` column in file {human_process_file_path} 
                and merge it into file {data_file_path}. If you are done, type 1 and press enter.
                """
            )
            if int(input()) != 1:
                exit()
        eval_result = Evaluator(
            **config,
            **eval_config['same'],
            **eval_config[config['dataset']]
        ).evaluate()
        eval_result_file_path = build_path(
            config['output_dir'], config['template_type'], config['template_idx'],
            tst_type, config['dataset'], str(Task.E), f'{llm_type.type}-eval.json'
        )

        os.makedirs(os.path.dirname(eval_result_file_path), exist_ok=True)
        with open(eval_result_file_path, 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=4)

        print(
            'The result of rewriting {} on dataset {} '
            'is evaluated and saved at {}'.format(str(llm_type), config['dataset'], eval_result_file_path)
        )

    print('finished !')


if __name__ == '__main__':
    config_file_path = 'config/main.yaml'
    main(OmegaConf.load(config_file_path))
