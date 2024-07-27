更新记录

- 2024-7-13，初始化项目
- 2024-7-14
  - 配置从`ArgumentParser`更改为 yaml
  - 删除scripts
  - 删除`prepares`, `evaluators`的重复代码
  - 支持训练，以及使用plm分类器
- 2024-7-27, 添加bert acc结果

# 0 前言

本工作的核心目标在于评估大型语言模型在文本风格转换任务中的效能，并将其与先前小型参数模型的表现进行对比分析。鉴于此，本研究中所选用的小型参数模型所产生的结果均来源于可公开访问的代码库，规避复现中的各式各样的问题。下列表格详列了本次研究所采纳的小型参数模型生成结果的具体出处。

*提示：这里你也发现了，每个数据集具体有多少数据参与了大模型测试，却决于小模型公布了多少改写结果。比如 Gender和 Political，基线仓库里就公布了1k条数据的改写结果，所以在这两个数据集上的大模型测试数据量是1k。*

**基线模型结果来源**


| 数据集       | 模型                                                                      | 仓库                                                                                                 |
| ------------ | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Yelp         | TSST                                                                      | [Link](https://github.com/xiaofei05/TSST/tree/master/outputs/yelp)                                   |
|              | CorssAligned, StyleEmbedding, MultiDecoder, D&R, DualRL, B-GST, G-GST     | [Link](https://github.com/rungjoo/Stable-Style-Transformer/tree/master/evaluation/yelp/compare/yelp) |
|              | StyIns                                                                    | [Link](https://github.com/XiaoyuanYi/StyIns/tree/master/styins_outputs/yelp)                         |
|              | IMaT                                                                      | [Link](https://github.com/zhijing-jin/IMaT/tree/master/outputs/yelp)                                 |
|              | StyleTransformer                                                          | [Link](https://github.com/fastnlp/style-transformer/tree/master/outputs/yelp)                        |
| ImageCaption | CorssAligned, StyleEmbedding, MultiDecoder, DeleteOnly, D&R, B-GST, G-GST | [Link](https://github.com/nnnngo/transformer-drg-style-transfer/tree/master/results/imagecaption)    |
| Gender       | B-GST                                                                     | [Link](https://github.com/nnnngo/transformer-drg-style-transfer/tree/master/results/gender)          |
| Political    | BackTranslation, B-GST, G-GST                                             | [Link](https://github.com/nnnngo/transformer-drg-style-transfer/tree/master/results/political)       |

# 1 项目介绍

项目总体结构分为四部分：① 评估器准备；② 大模型改写（调接口或自行部署）；③改写结果处理；④ 评估改写结果

## 1.1 评估器准备部分介绍

**保存点下载链接：[Link](https://pan.baidu.com/s/1K3m-k_henrQTIzYmZXKA4Q?pwd=1234 )**

- 内容保留度通过BLEU衡量
- 风格迁移能力借助风格分类器衡量
- 流畅度评分采用 [GPT-2-large](https://huggingface.co/openai-community/gpt2-large) 以及在数据集上微调后的 [GPT-2-small](https://huggingface.co/openai-community/gpt2) 衡量

因此，需要提前准备好的评估器有 **分类器** 以及 微调后的 **GPT-2-small**

### 1.1.1目录结构

```
|-- prepare.py				程序主入口
|-- prepares/
|   |-- __init__.py
|   |-- base_classifier.py  训练分类器
|   |-- base_fluency.py     训练GPT2
|   |-- dataset.py			定义数据集类
|   `-- default.py			数据集虽然不同，但是训练方式相同，训练代码都提出来放在了该文件中
```

### 1.1.2 准备分类器

配置文件：`config/prepare.yaml`

```shell
# 修改配置文件的task为classifier
python prepare.py
```

训练的fasttex分类器，相关训练参数和结果如下


| 数据集       | epoch | lr  | loss | wordNgrams | acc on valid |
| ------------ | ----- | --- | ---- | ---------- | ------------ |
| yelp         | 35    | 1   | hs   | 2          | 0.973        |
| amazon       | 35    | 1.4 | hs   | 2          | 0.808        |
| imagecaption | 305   | 1.2 | hs   | 3          | 0.772        |
| gender       | 5     | 1   | hs   | 2          | 0.824        |
| political    | 25    | 1.3 | hs   | 4          | 0.830        |


你也可以使用在数据集上微调后的PLM作为分类器。只需要将`config/prepare.yaml`中的`model_type`的值修改为 *plm* 即可，训练相关设置在`train_config.classifier.plm`。部分数据集的分类器在HuggingFace上有现成的。

| 数据集       | 模型       | Acc [dev/test] | 链接                                                         |
| ------------ | ---------- | -------------- | ------------------------------------------------------------ |
| yelp         | BERT       | 98.25/98.6     | [BaiduNetDisk Link](https://pan.baidu.com/s/1K3m-k_henrQTIzYmZXKA4Q?pwd=1234 ) |
| amazon       | BERT       | -/94.65        | [Huggingface Link](https://huggingface.co/fabriceyhc/bert-base-uncased-amazon_polarity) |
| imagecaption | BERT       | 82.19/-        | [BaiduNetDisk Link](https://pan.baidu.com/s/1K3m-k_henrQTIzYmZXKA4Q?pwd=1234 ) |
| gender       | DistilBERT | -/1.0          | [Huggingface Link](https://huggingface.co/padmajabfrl/Gender-Classification/tree/main) |
| political    | BERT       | -/0.8393       | [HuggingfaceLink](https://huggingface.co/harshal-11/Bert-political-classification/tree/main) |

### 1.1.3 准备微调后的GPT-2 small

配置文件：`config/prepare.yaml`

```shell
# 修改配置文件的task为fluency
python prepare.py
```

| 数据集       | ppl on valid |
| ------------ | ------------ |
| yelp         | 14.24        |
| amazon       | 23.78        |
| imagecaption | 29.51        |
| gender       | 17.02        |
| political    | 29.61        |

## 1.2 大模型改写&后处理&评估

**基础说明**

| 项目         | 文件                                                         | 备注                                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 数据下载地址 | [Link](https://pan.baidu.com/s/1K3m-k_henrQTIzYmZXKA4Q?pwd=1234) |                                                              |
| 配置文件     | `config/main.yaml`                                           | 一共三个阶段：改写，处理，评估，通过配置可以指定执行哪个阶段<br/>改写阶段，你需要修改配置文件中的`rewrite_config`<br/>评估阶段，你需要修改配置文件中的`eval_config` |
| 运行         | `python main.py`                                             |                                                              |

**其他说明：**

| 阶段 | 输入文件                                                     | 输出文件                                                     |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 改写 | 格式：`{data_dir}/{tst_type}/{dataset name}/{split}.csv`<br/>例子：`data/pos-neg/yelp/test.csv`，其中`tst_type`取值范围见`global_config.py`中的`DATASET_TO_TST_TYPE` | 格式：`{output_dir}/{template_type}/{template_idx}/{tst_type}/{dataset name}/rewrite/{llm_type}.csv`<br/>例子：`output/common/0/pos-neg/yelp/rewrite/qwen-7b-chat.csv` |
| 处理 | 上一步的输出文件                                             | 例子：`output/common/0/pos-neg/process/qwen-7b-chat-processed.csv` |
| 评估 | 上一步的输出文件                                             | 例子：`output/common/0/pos-neg/evaluate/qwen-7b-chat-eval.json` |

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

## 1.3 其他文件


| 文件/目录   | 说明                   |
| ----------- | ---------------------- |
| baselines   | 所有的基线模型数据     |
| baseline.py | 评估基线模型改写的结果 |
| constant.py | 定义的常量             |
| utils.py    | 封装的一些工具         |
