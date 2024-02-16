# 1 Basic Information

The primary objective of this paper is to assess the performance of LLMs in text style transfer task. To achieve this, we replicate all the evaluated models from previous research to ensure we can compare our experimental results. Additionally, we provide details about the datasets used in the paper, as well as the sources of baseline results.

**Dataset Information**

| Dataset      | S ↔ T                                   | Split                                          | Description                                                  | Source                                                       |
| ------------ | --------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Yelp         | Positive(1) ↔ Negative(0)               | train: 443,259<br/>dev: 4,000<br/>test: 1,000  |                                                              | \[[Code](https://github.com/lijuncen/Sentiment-and-Style-Transfer)\]  [Paper[^3]] |
| Amazon       | Positive(1) ↔ Negative(0)               | train: 554,997<br/>dev: 2,000<br/>test: 1,000  |                                                              | \[[Code](https://github.com/lijuncen/Sentiment-and-Style-Transfer)\]  [Paper[^3]] |
| Imagecaption | Factual(2) → Humorous(0) or Romantic(1) | train: 12,000<br/>dev: 1,000<br/>test: 300     | Factual doesn’t exist in train and dev<br/>Only factual in the test | \[[Code](https://github.com/lijuncen/Sentiment-and-Style-Transfer)\]  [Paper[^3]] |
| Political    | Republican(0) ↔ Democratic(1)           | train: 537,922<br/>dev: 4,000<br/>test: 56,000 |                                                              | [Paper[^5][^6]]<br/>[[Code](https://github.com/shrimai/Style-Transfer-Through-Back-Translation)]<br/>[[Link](http://tts.speech.cs.cmu.edu/style_models/political_data.tar)] |
| Gender       | Male(0) ↔ Female(1)                     | train: 200,000<br/>dev: 2,000<br/>test: 4,000  |                                                              | [Paper[^5][^7]<br/>[[Code](https://github.com/EwoeT/MLM-style-transfer/)]<br/>[[Link](http://tts.speech.cs.cmu.edu/style_models/gender_data.tar)] |

**Source of baseline results**

| Source                                                       | Dataset      | Metric                                                       |
| ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ |
| [Paper[^1]]<br/>[[Code](https://github.com/xiaofei05/TSST)]  | Yelp         | Acc: fine-tuning bert-base on train set (98.8% on test)<br>s-BLEU: corpus-bleu<br/>r-BLEU: corpus-bleu<br>PPL: KenLM<br>GM: $\sqrt[4]{\frac{\text{self-BLEU}\times \text{ref-BLEU} \times \text{acc}}{\text{log}_{\text{PPL}}}}$ |
| [Paper[^2]]<br/>[[Code](https://github.com/rungjoo/Stable-Style-Transformer)] | Amazon       | Acc: fasttext<br/>s-BLEU <br/>r-BLEU<br/>g-BLEU: $\sqrt{\text{s-BLEU}\times \text{h-BLEU}}$<br/>d-PPL: fine-tuning gpt2 on train set (ppl=24.68 on dev)<br/>g-PPL: gpt2-large<br/>t-PPL: $\sqrt{\text{d-PPL}\times \text{g-PPL}}$<br/>BERTscore: roberta-large |
| [Paper[^4]]<br/>[[Code](https://github.com/agaralabs/transformer-drg-style-transfer)] | Imagecaption | Acc: FastText (acc=77.2%)<br/>s-BLEU<br/>h-BLEU<br/>PPL: fint-tuning gpt2-small on train set (ppl = 29.51 on dev set) |
| [Paper[^4]<br/>[[Code](https://github.com/agaralabs/transformer-drg-style-transfer)] | Male-Female  | Acc: FastText(acc=82.4% on dev)<br/>PPL: fine-tuning gpt2-small (ppl=17.02 on dev)<br/>s-BLEU |
| [Paper[^4]]<br/>[[Code](https://github.com/agaralabs/transformer-drg-style-transfer)] | Political    | Acc: FastText(==acc=81.7% on dev==)<br/>PPL: fine-tuning gpt2-small (ppl=29.61 on dev)<br/>s-BLEU |

# 2 Usage

## 2.1 How to prepare the evaluation models?

```shell
python prepare.py \
--dataset yelp \
--task classifier \
--pretrained_dir_or_name ./pretrained_models/bert-base
```

The possible options for the dataset are *yelp*, *amazon*, *imagecaption*, *paper*, *political*, and *gender*.

Some datasets use pre-trained language models when evaluating ==acc== and ==ppl==, so you need to set the `pretrained_dir_or_name`.

The possible options for the task are *classifier* and *fluency*. It is recommended to look at the `DATASET_TO_FUN` variable in `prepare.py` to be informed of the types of tasks that this dataset can have.

## 2.2 How to rewrite text by LLMs











[^1]: Fei Xiao, Liang Pang, Yanyan Lan, Yan Wang, Huawei Shen, and Xueqi Cheng. 2021. [Transductive Learning for Unsupervised Text Style Transfer](https://aclanthology.org/2021.emnlp-main.195). In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 2510-2521, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

[^2]: Joosung Lee. 2020. [Stable Style Transformer: Delete and Generate Approach with Encoder-Decoder for Text Style Transfer](https://aclanthology.org/2020.inlg-1.25). In *Proceedings of the 13th International Conference on Natural Language Generation*, pages 195–204, Dublin, Ireland. Association for Computational Linguistics.

[^3]: Juncen Li, Robin Jia, He He, and Percy Liang. 2018. [Delete, Retrieve, Generate: a Simple Approach to Sentiment and Style Transfer](https://aclanthology.org/N18-1169). In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pages 1865–1874, New Orleans, Louisiana. Association for Computational Linguistics.
[^4]: Akhilesh Sudhakar, Bhargav Upadhyay, and Arjun Maheswaran. 2019. [“Transforming” Delete, Retrieve, Generate Approach for Controlled Text Style Transfer](https://aclanthology.org/D19-1322). In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 3269–3279, Hong Kong, China. Association for Computational Linguistics.
[^5]:Shrimai Prabhumoye, Yulia Tsvetkov, Ruslan Salakhutdinov, and Alan W Black. 2018. [Style Transfer Through Back-Translation](https://aclanthology.org/P18-1080). In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 866–876, Melbourne, Australia. Association for Computational Linguistics.

[^6]:Rob Voigt, David Jurgens, Vinodkumar Prabhakaran, Dan Jurafsky, and Yulia Tsvetkov. 2018. [RtGender: A Corpus for Studying Differential Responses to Gender](https://aclanthology.org/L18-1445). In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*, Miyazaki, Japan. European Language Resources Association (ELRA).
[^7]: Sravana Reddy and Kevin Knight. 2016. [Obfuscating Gender in Social Media Writing](https://aclanthology.org/W16-5603). In *Proceedings of the First Workshop on NLP and Computational Social Science*, pages 17–26, Austin, Texas. Association for Computational Linguistics.
