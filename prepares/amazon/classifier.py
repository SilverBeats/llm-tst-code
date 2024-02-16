from pprint import pp

from utils import build_path
from ..default import train_fasttext_classifier


def main(
        data_dir: str,
        output_dir: str,
        **kwargs
):
    train_file_path = build_path(data_dir, 'fasttext.train')
    dev_file_path = build_path(data_dir, 'dev.csv')
    ckpt_file_path = build_path(output_dir, 'classifier.pk')
    train_args = {
        'epoch': 35,
        'lr': 1.4,
        'loss': 'hs',
        'wordNgrams': 2
    }
    pp(train_args)
    train_fasttext_classifier(
        train_file_path=train_file_path,
        ckpt_file_path=ckpt_file_path,
        dev_file_path=dev_file_path,
        output_dir=output_dir,
        train_args=train_args
    )
