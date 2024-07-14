from constant import AccModelType
from .default import train_fasttext_classifier, train_classifier_bert


def main(
        model_type: AccModelType,
        output_dir: str,
        data_dir: str,
        device: str,
        seed: int,
        train_config: dict = None,
        **kwargs
):
    if train_config is None:
        train_config = {}
    if model_type == AccModelType.FASTTEXT:
        train_fasttext_classifier(
            data_dir=data_dir,
            output_dir=output_dir,
            seed=seed,
            **train_config
        )
    elif model_type == AccModelType.PLM:
        train_classifier_bert(
            data_dir=data_dir,
            output_dir=output_dir,
            seed=seed,
            device=device,
            **train_config
        )
