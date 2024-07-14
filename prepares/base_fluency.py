from constant import FluencyModelType
from .default import train_fluency_gpt2


def main(
        model_type: FluencyModelType,
        output_dir: str,
        data_dir: str,
        device: str,
        seed: int,
        train_config: dict = None,
        **kwargs
):
    if train_config is None:
        train_config = {}
    if model_type == FluencyModelType.PLM:
        train_fluency_gpt2(
            data_dir=data_dir,
            output_dir=output_dir,
            seed=seed,
            device=device,
            **train_config
        )
