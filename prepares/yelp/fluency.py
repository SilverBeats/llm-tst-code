from ..default import train_fluency_gpt2


def main(
        pretrained_dir_or_name: str,
        output_dir: str,
        data_dir: str,
        device: str,
        seed: int,
        **kwargs
):
    train_fluency_gpt2(
        pretrained_model_name_or_dir=pretrained_dir_or_name,
        data_dir=data_dir,
        output_dir=output_dir,
        seed=seed,
        device=device,
        do_test=True
    )
