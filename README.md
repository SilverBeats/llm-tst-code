*Exploring Large Language Models Text Style Transfer Capabilities* (Accepted By ECAI 2024)

# 0 前言

本工作的核心目标在于评估大型语言模型在文本风格转换任务中的效能，并将其与先前小型参数模型的表现进行对比分析。鉴于此，本研究中所选用的小型参数模型所产生的结果均来源于可公开访问的代码库，规避复现中的各式各样的问题。下列表格详列了本次研究所采纳的小型参数模型生成结果的具体出处。

*提示：这里你也发现了，每个数据集具体有多少数据参与了大模型测试，却决于小模型公布了多少改写结果。比如 Gender和 Political，基线仓库里就公布了1k条数据的改写结果，所以在这两个数据集上的大模型测试数据量是1k。*

**基线模型结果来源**

| 数据集       | 模型                                                         | 仓库                                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Yelp         | TSST                                                         | [Link](https://github.com/xiaofei05/TSST/tree/master/outputs/yelp) |
|              | CorssAligned, StyleEmbedding, MultiDecoder, D&R, DualRL, B-GST, G-GST | [Link](https://github.com/rungjoo/Stable-Style-Transformer/tree/master/evaluation/yelp/compare/yelp) |
|              | StyIns                                                       | [Link](https://github.com/XiaoyuanYi/StyIns/tree/master/styins_outputs/yelp) |
|              | IMaT                                                         | [Link](https://github.com/zhijing-jin/IMaT/tree/master/outputs/yelp) |
|              | StyleTransformer                                             | [Link](https://github.com/fastnlp/style-transformer/tree/master/outputs/yelp) |
| ImageCaption | CorssAligned, StyleEmbedding, MultiDecoder, DeleteOnly, D&R, B-GST, G-GST | [Link](https://github.com/nnnngo/transformer-drg-style-transfer/tree/master/results/imagecaption) |
| Gender       | B-GST                                                        | [Link](https://github.com/nnnngo/transformer-drg-style-transfer/tree/master/results/gender) |
| Political    | BackTranslation, B-GST, G-GST                                | [Link](https://github.com/nnnngo/transformer-drg-style-transfer/tree/master/results/political) |

# 1 项目介绍

项目总体结构分为四部分：① 评估器准备；② 大模型改写（调接口或自行部署）；③改写结果处理；④ 评估改写结果 

## 1.1 评估器准备部分介绍

**保存点下载链接：[Link](https://pan.baidu.com/s/1K3m-k_henrQTIzYmZXKA4Q?pwd=1234 )**

-   内容保留度通过BLEU衡量
-   风格迁移能力借助风格分类器衡量
-   流畅度评分采用 [GPT-2-large](https://huggingface.co/openai-community/gpt2-large) 以及在数据集上微调后的 [GPT-2-small](https://huggingface.co/openai-community/gpt2) 衡量

因此，需要提前准备好的评估器有 **分类器** 以及 微调后的 **GPT-2-small**

### 1.1.1目录结构

```
|-- prepare.py				程序主入口
|-- prepares/
|   |-- __init__.py
|   |-- yelp/				每个数据集名称目录下都会有两个文件，对应：训练分类器、训练GPT-2
|   |   |-- __init__.py
|   |   |-- classifier.py
|   |   `-- fluency.py
|   |-- captions/
|   |-- gender/
|   |-- political/
|   |-- amazon/
|   |-- dataset.py			定义数据集类
|   `-- default.py			数据集虽然不同，但是训练方式相同，训练代码都提出来放在了该文件中
```

### 1.1.2 准备分类器

```shell
python prepare.py --dataset yelp --task classifier
```

训练的fasttex分类器，相关训练参数和结果如下

| 数据集       | epoch | lr   | loss | wordNgrams | acc on valid |
| ------------ | ----- | ---- | ---- | ---------- | ------------ |
| yelp         | 35    | 1    | hs   | 2          | 0.973        |
| amazon       | 35    | 1.4  | hs   | 2          | 0.808        |
| imagecaption | 305   | 1.2  | hs   | 3          | 0.772        |
| gender       | 5     | 1    | hs   | 2          | 0.824        |
| political    | 25    | 1.3  | hs   | 4          | 0.830        |

### 1.1.3 准备微调后的GPT-2 small

```shell
python prepare.py -dataset yelp --task fluency
```

训练参数见`default.py/train_fluency_gpt2`

| 数据集       | ppl on valid |
| ------------ | ------------ |
| yelp         | 14.24        |
| amazon       | 23.78        |
| imagecaption | 29.51        |
| gender       | 17.02        |
| political    | 29.61        |

## 1.2 大模型改写

**数据下载地址：[Link](https://pan.baidu.com/s/1K3m-k_henrQTIzYmZXKA4Q?pwd=1234)**

### 1.2.1 目录结构

```
|-- global_config.py			配置信息，比如模板、api等		
|-- main.py						程序主入口
|-- rewriters/					
|   |-- __init__.py
|   |-- base.py					
|   |-- llama2.py				llama2的改写器
|   |-- mistral.py				mistral的改写器
|   |-- openai.py				GPT的改写器
|   |-- qwen.py					qwen的改写器
|   `-- rewrite.py				对上述四款改写器做的封装，以模型名称为参数，创建不同的改写器
```

### 1.2.2 运行

```shell
python main.py\
	--dataset yelp\
	--llm_type qwen-7b-chat\
	--do_rewrite
	# 如果是本地部署的llm，比如llama或者mistral，你需要通过这个参数给出llm文件的路径
	# --llm_model_dir 'xxxx' 
```

默认加载的数据文件路径：

-   格式：`{data_dir}/{tst_type}/{dataset name}/{split}.csv`
-   例子：`data/pos-neg/yelp/test.csv`，其中`tst_type`取值范围见`global_config.py`中的`DATASET_TO_TST_TYPE`

默认输出目录：

-   格式：`{output_dir}/{template_type}/{template_idx}/{tst_type}/{dataset name}/rewrite/{llm_type}.csv`
-   例子：`output/common/0/pos-neg/yelp/rewrite/qwen-7b-chat.csv`

## 1.3 处理改写数据

### 1.3.1 目录结构

```
|-- global_config.py		同上
|-- main.py					同上
|-- processors/	
|   |-- __init__.py
|   |-- processor.py		以模型名称为参数，调用utils.py中不同的处理方法
|   `-- utils.py			针对不同模型封装的处理代码
```

### 1.3.2 运行

```shell
python main.py\
	--dataset yelp\
	--llm_type qwen-7b-chat\
	--do_process
```

加载的数据文件：上一步改写的输出文件

输出目录：`output/common/0/pos-neg/process/qwen-7b-chat-processed.csv`

## 1.4 评估

### 1.4.1 目录结构

```
|-- evaluators/				不同数据集有其自己的评估类，但基本上都一样，所以在base.py中有个父类
|   |-- __init__.py
|   |-- amazon.py
|   |-- base.py				
|   |-- caption.py
|   |-- evaluator.py		以数据集名称为参数，创建不同的评估器
|   |-- gender.py
|   |-- political.py
|   |-- utils.py
|   `-- yelp.py
|-- global_config.py		同上
|-- main.py					同上
```

### 1.4.2 运行

```shell
python main.py\
	--dataset yelp\
	--llm_type qwen-7b-chat\
	--do_eval\
	--acc_model_dir_or_path 'fastext保存点路径'
	--d_ppl_model_dir '微调过的gpt2 small路径'
	--bert_score_model_dir '计算bertscore用到的plm文件路径'
	--bert_score_model_layers '计算bertscore用到的plm的层数'
```

加载的数据文件：process后的文件

输出目录：`output/common/0/pos-neg/evaluate/qwen-7b-chat-eval.json`



最终一套下来你会得到如下的目录输出结构

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

## 1.5 其他文件

| 文件/目录   | 说明                                            |
| ----------- | ----------------------------------------------- |
| baselines   | 所有的基线模型数据                              |
| baseline.py | 评估基线模型改写的结果                          |
| scripts     | 写好的脚本，可以通过该脚本去掌握`main.py`的用法 |
| utils.py    | 封装的一些工具                                  |

