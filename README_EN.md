# 0 Preamble

The central aim of this endeavor is to appraise the capability of large language models in textual style transfer tasks and juxtapose their performance with that of preceding models with fewer parameters. To accomplish this, the outcomes from the latter models employed herein are derived solely from code repositories that are openly available, thus sidestepping an array of potential issues that might surface during the replication phase. The subsequent table scrupulously enumerates the precise sources of the generated results from the smaller models encompassed in this investigation.

*Note: Here, you may notice that the quantity of data involved in the testing of LLMs for each dataset is contingent upon the number of rewriting results disclosed by the small models. For instance, in the case of Gender and Political datasets, the baseline repository has published rewriting results for 1,000 pieces of data; hence, the volume of data used for testing large models on these datasets stands at 1,000.*

**Note:**  You may notice here that the amount of data each dataset contributes to the LLM's testing phase is contingent upon the number of rewritten outputs shared by the small-scale models. For instance, in the case of the *Gender* and *Political* datasets, the baseline repository discloses 1,000 rewritten samples, hence the volume of data for the LLM's evaluation on these two datasets stands at 1,000 entries.


**Source of small-scale model style transfer results**
| Dataset       | Model                                                         | Repo Link                                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Yelp         | TSST                                                         | [Link](https://github.com/xiaofei05/TSST/tree/master/outputs/yelp) |
|              | CorssAligned, StyleEmbedding, MultiDecoder, D&R, DualRL, B-GST, G-GST | [Link](https://github.com/rungjoo/Stable-Style-Transformer/tree/master/evaluation/yelp/compare/yelp) |
|              | StyIns                                                       | [Link](https://github.com/XiaoyuanYi/StyIns/tree/master/styins_outputs/yelp) |
|              | IMaT                                                         | [Link](https://github.com/zhijing-jin/IMaT/tree/master/outputs/yelp) |
|              | StyleTransformer                                             | [Link](https://github.com/fastnlp/style-transformer/tree/master/outputs/yelp) |
| ImageCaption | CorssAligned, StyleEmbedding, MultiDecoder, DeleteOnly, D&R, B-GST, G-GST | [Link](https://github.com/nnnngo/transformer-drg-style-transfer/tree/master/results/imagecaption) |
| Gender       | B-GST                                                        | [Link](https://github.com/nnnngo/transformer-drg-style-transfer/tree/master/results/gender) |
| Political    | BackTranslation, B-GST, G-GST                                | [Link](https://github.com/nnnngo/transformer-drg-style-transfer/tree/master/results/political) |

# 1 Project Introduction

The overarching framework of the project is segmented into four principal components:

1.  Preparations for the evaluator;
2.  Rewrite operations executed by the LLMs;
3.  Processing of the resultant rewrite outputs;
4.  Evaluation of these outputs post-rewrite.

## 1.1 Prepare the Evaluators

**Click [here](https://pan.baidu.com/s/1K3m-k_henrQTIzYmZXKA4Q?pwd=1234 ) to download the checkpoints**

- Content retention is measured by BLEU
- Style transfer ability is measured with style classifiers
- Fluency is measured with [GPT2-large](https://huggingface.co/openai-community/gpt2-large) and fine-tuned [GPT2-small](https://huggingface.co/openai-community/gpt2)

Therefore, the evaluators that need to be prepared in advance are the **classifier** and the **fine-tuned GPT-2-small**

### 1.1.1 Directory Structure

```
|-- prepare.py				program main entry
|-- prepares/
|   |-- __init__.py
|   |-- yelp/				There will be two files under each dataset directory, 														corresponding to: Training Classifier, Training GPT-2
|   |   |-- __init__.py
|   |   |-- classifier.py
|   |   `-- fluency.py
|   |-- captions/
|   |-- gender/
|   |-- political/
|   |-- amazon/
|   |-- dataset.py			Defining a dataset class
|   `-- default.py			Although the datasets are different, the training methods 							  						are the same, and the training codes are all
							proposed and placed in this file
```

### 1.1.2 Train classifier

```shell
python prepare.py --dataset yelp --task classifier
```

The particulars alongside the results of the training process for the FastText classifier are delineated as follows:

| Dataset      | epoch | lr   | loss | wordNgrams | Acc on valid |
| ------------ | ----- | ---- | ---- | ---------- | ------------ |
| yelp         | 35    | 1    | hs   | 2          | 0.973        |
| amazon       | 35    | 1.4  | hs   | 2          | 0.808        |
| imagecaption | 305   | 1.2  | hs   | 3          | 0.772        |
| gender       | 5     | 1    | hs   | 2          | 0.824        |
| political    | 25    | 1.3  | hs   | 4          | 0.830        |

### 1.1.3 Train GPT2-small

```shell
python prepare.py --dataset yelp --task fluency
```

For training parameters, see `default.py/train_fluency_gpt2`

| Dataset      | PPL on valid |
| ------------ | ------------ |
| yelp         | 14.24        |
| amazon       | 23.78        |
| imagecaption | 29.51        |
| gender       | 17.02        |
| political    | 29.61        |

## 1.2 Rewrite

**Click [here](https://pan.baidu.com/s/1K3m-k_henrQTIzYmZXKA4Q?pwd=1234) to download the dataset**

### 1.2.1 Directory Structure

```
|-- global_config.py			Configuration, including prompt template, api, etc.		
|-- main.py						program main entry
|-- rewriters/					
|   |-- __init__.py
|   |-- base.py					
|   |-- llama2.py				llama2’s rewriter
|   |-- mistral.py				mistral’s rewriter
|   |-- openai.py				GPT’s rewriter
|   |-- qwen.py					qwen’s rewriter
|   `-- rewrite.py				Create a different rewriter, taking the model name as an argument
```

### 1.2.2 Run

```shell
python main.py\
	--dataset yelp\
	--llm_type qwen-7b-chat\
	--do_rewrite
	# If the llm is locally deployed, such as llama or mistral, 
	# you need to specify the path of the llm file with this paramet
	# --llm_model_dir 'xxxx' 
```

Default loaded data file path:

-   Format：`{data_dir}/{tst_type}/{dataset name}/{split}.csv`
-   E.g.：`data/pos-neg/yelp/test.csv`，For the value range of `tst type`, see `DATASET_TO_TST_TYPE` in `global config.py`

Default output directory：

-   Format：`{output_dir}/{template_type}/{template_idx}/{tst_type}/{dataset name}/rewrite/{llm_type}.csv`
-   E.g.：`output/common/0/pos-neg/yelp/rewrite/qwen-7b-chat.csv`

## 1.3 Postprocessing

### 1.3.1 Directory Structure

```
|-- global_config.py		omitted
|-- main.py					omitted
|-- processors/	
|   |-- __init__.py
|   |-- processor.py		Different processing methods in utils.py are called, 
							taking the model name as an argument
|   `-- utils.py			Processing code encapsulated for different models
```

### 1.3.2 Run

```shell
python main.py\
	--dataset yelp\
	--llm_type qwen-7b-chat\
	--do_process
```

Loaded data file: Output file rewritten in the previous step

Output directory: `output/common/0/pos-neg/process/qwen-7b-chat-processed.csv`

## 1.4 Evaluation

### 1.4.1 Directory Structure

```
|-- evaluators/
|   |-- __init__.py
|   |-- amazon.py
|   |-- base.py				
|   |-- caption.py
|   |-- evaluator.py		Create different evaluators with the data set name as a parameter
|   |-- gender.py
|   |-- political.py
|   |-- utils.py
|   `-- yelp.py
|-- global_config.py		omitted
|-- main.py					omitted
```

### 1.4.2 Run

```shell
python main.py\
	--dataset yelp\
	--llm_type qwen-7b-chat\
	--do_eval\
	--acc_model_dir_or_path "fastext's checkpoint path"
	--d_ppl_model_dir "fine-tuned GPT2-small path"
	--bert_score_model_dir "plm path for bertscore"
	--bert_score_model_layers 'the plm layer num for bertscore'
```

Loaded data file: The file after post-processing

Output directory：`output/common/0/pos-neg/evaluate/qwen-7b-chat-eval.json`



You end up with the following directory output structure

```
|-- output/common/0/pos-neg/yelp
|	|--rewrite/
|	|	|-- qwen-7b-chat.csv
|	|	|-- qwen-14b-chat.csv
|	|--process/
|	|	|-- qwen-7b-chat-processed.csv
|	|	|-- qwen-14b-chat-processed.csv
|	|--evaluate/
|	|	|-- qwen-7b-chat-eval.json
|	|	|-- qwen-14b-chat-eval.json
```

## 1.5 Other files

| File/Directory | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| baselines      | All baseline model data                                      |
| baseline.py    | Evaluate the results of baseline model rewriting             |
| scripts        | The written script can be used to master the usage of `main.py` |
| utils.py       | Encapsulate some of the tools                                |