from .base import HasOutputEvaluator


class AmazonEvaluator(HasOutputEvaluator):
    DATASET = 'amazon'

    def __init__(
            self,
            data_file_path: str,
            ref_file_path: str,
            acc_model_dir_or_path: str,
            d_ppl_model_dir: str,
            g_ppl_model_dir: str,
            bert_score_model_dir: str,
            bert_score_model_layers: int,
            device: str,
            batch_size: int,
            **kwargs
    ):
        super().__init__(
            data_file_path=data_file_path,
            acc_model_dir_or_path=acc_model_dir_or_path,
            d_ppl_model_dir=d_ppl_model_dir,
            g_ppl_model_dir=g_ppl_model_dir,
            bert_score_model_dir=bert_score_model_dir,
            bert_score_model_layers=bert_score_model_layers,
            ref_file_path=ref_file_path,
            device=device,
            batch_size=batch_size,
            **kwargs
        )
