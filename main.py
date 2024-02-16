import os
import warnings
from argparse import ArgumentParser
from pprint import pp
from typing import Any, Dict
from rewriters import Rewriter
from evaluators import Evaluator
from processors import Processor
from global_config import ACCEPTABLE_DATASET, ACCEPTABLE_LLM, DATASET_TO_TST_TYPE, Task
from utils import build_path, get_LLMType


def get_args() -> Dict[str, Any]:
    parser = ArgumentParser()
    # must be given
    parser.add_argument('--dataset', type=str, required=True, choices=ACCEPTABLE_DATASET)
    parser.add_argument('--llm_type', type=get_LLMType, required=True, choices=ACCEPTABLE_LLM)

    parser.add_argument('--do_rewrite', action='store_true')
    parser.add_argument('--do_process', action='store_true')
    parser.add_argument('--do_eval', action='store_true')

    parser.add_argument('--template_type', type=str, default='common', choices=['common', 'special'])
    parser.add_argument('--template_idx', type=int, default=0)

    # not often to edit
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=8)

    # will edit when do not request api to do rewrite (like falcon, mistral, llama2)
    parser.add_argument('--llm_model_dir', type=str, default=None,
                        help='the large language model dir. For loading model to do text style transfer.')

    # used when do eval
    parser.add_argument('--acc_model_dir_or_path', type=str, default=None)
    parser.add_argument('--d_ppl_model_dir', type=str, default=None)
    parser.add_argument('--g_ppl_model_dir', type=str, default='pretrained_models/gpt2-large')
    parser.add_argument('--bert_score_model_dir', type=str, default='pretrained_models/roberta-large')
    parser.add_argument('--bert_score_model_layers', type=int, default=17)

    return vars(parser.parse_args())


def main():
    config = get_args()
    pp(config)
    llm_type = config['llm_type']
    do_rewrite, do_process, do_eval = config.pop('do_rewrite'), config.pop('do_process'), config.pop('do_eval')

    if do_rewrite:
        Rewriter(**config).rewrite()
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
        Evaluator(**config).evaluate()
        print(
            'The result of rewriting {} on dataset {} '
            'is evaluated.'.format(str(llm_type), config['dataset'])
        )

    print('finished !')


if __name__ == '__main__':
    main()
