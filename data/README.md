# Positive-Negative

| Dataset | Link                                                         | Label             | size                                              | parallel |
| ------- | ------------------------------------------------------------ | ----------------- | ------------------------------------------------- | -------- |
| yelp    | [Download](https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data) | 0: Neg<br/>1: Pos | train: 443,259 <br/>valid: 4,000 <br/>test: 1,000 | No       |
| amazon  | [Download](https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data) | 0: Neg<br/>1: Pos | train: 554,997<br/>valid: 2,000<br/>test: 1,000   | No       |
| imdb    | [Download](https://huggingface.co/datasets/imdb)             | 0: Neg<br/>1: Pos | train: 25,000<br/>test: 25,000                  | No       |

```
##############################yelp, amazon##############################
@inproceedings{li-etal-2018-delete,
    title = "Delete, Retrieve, Generate: a Simple Approach to Sentiment and Style Transfer",
    author = "Li, Juncen  and
      Jia, Robin  and
      He, He  and
      Liang, Percy",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-1169",
    doi = "10.18653/v1/N18-1169",
    pages = "1865--1874"
}

##############################imdb##############################
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```

# Humorous-Romantic

| Dataset      | Link                                                         | Label                       | size                                           | parallel |
| ------------ | ------------------------------------------------------------ | --------------------------- | ---------------------------------------------- | -------- |
| imagecaption | [Download](https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data) | 0: humorous<br/>1: romantic | train: 12,000 <br/>valid: 1,000 <br/>test: 600 | No       |

```
@inproceedings{li-etal-2018-delete,
    title = "Delete, Retrieve, Generate: a Simple Approach to Sentiment and Style Transfer",
    author = "Li, Juncen  and
      Jia, Robin  and
      He, He  and
      Liang, Percy",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-1169",
    doi = "10.18653/v1/N18-1169",
    pages = "1865--1874"
}
```

# Paper-News

| Dataset | Link                                                         | Label                | size                                          | parallel |
| ------- | ------------------------------------------------------------ | -------------------- | --------------------------------------------- | -------- |
| Paper   | [Download](https://github.com/fuzhenxin/textstyletransferdata/blob/master/eval/train) | 0: Paper<br/>1: News | train: 205,006 <br/>dev: 2,000<br>test: 2,000 | No       |

```
@inproceedings{fu2018style,
  title={Style transfer in text: Exploration and evaluation},
  author={Fu, Zhenxin and Tan, Xiaoye and Peng, Nanyun and Zhao, Dongyan and Yan, Rui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={32},
  number={1},
  year={2018}
}
```

# Republican-Democratic

| Dataset   | Link                                                         | Label                           | size                                               | parallel |
|-----------| ------------------------------------------------------------ | ------------------------------- | -------------------------------------------------- | -------- |
| political | [Download](http://tts.speech.cs.cmu.edu/style_models/political_data.tar)<br/>[Github](https://github.com/shrimai/Style-Transfer-Through-Back-Translation) | 0: Republican<br/>1: Democratic | train: 537,922 <br/>valid: 4,000 <br/>test: 56,000 | No       |

```
@inproceedings{style_transfer_acl18,
    title={Style Transfer Through Back-Translation},
    author={Prabhumoye, Shrimai and Tsvetkov, Yulia and Salakhutdinov, Ruslan and Black, Alan W},
    year={2018},
    booktitle={Proc. ACL}
}
```

# Male-Female

| Dataset | Link                                                         | Label                 | size                                                | parallel |
|---------| ------------------------------------------------------------ | --------------------- | --------------------------------------------------- | -------- |
| gender  | [Download](http://tts.speech.cs.cmu.edu/style_models/gender_data.tar)<br/>[Github](https://github.com/shrimai/Style-Transfer-Through-Back-Translation) | 0: Male<br/>1: Female | train: 2,669,184<br/>valid: 4,492<br/>test: 534,460 | No       |

```
@inproceedings{rtgender,
    title={{RtGender}: A Corpus for Studying Differential Responses to Gender},
    author={Voigt, Rob and Jurgens, David and Prabhakaran, Vinodkumar and Jurafsky, Dan and Tsvetkov, Yulia},
    year={2018},
    booktitle={Proc. LREC},
}
```





