# EditSQL for [Spider](https://yale-lily.github.io/spider), [SParC](https://yale-lily.github.io/sparc), [CoSQL](https://yale-lily.github.io/cosql)

This is a pytorch implementation of the CD-Seq2Seq baseline and the EditSQL model in the following papers
- "[SParC: Cross-Domain Semantic Parsing in Context](https://arxiv.org/abs/1906.02285)", ACL 2019
- "[CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases](https://arxiv.org/pdf/1909.05378.pdf)", EMNLP 2019
- "[Editing-Based SQL Query Generation for Cross-Domain Context-Dependent Questions](https://arxiv.org/pdf/1909.00786.pdf)", EMNLP 2019

Please cite the papers if you use our data and code.
```
@InProceedings{yu2018spider,
    title = "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task",
    author = "Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, Dragomir Radev",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    year = "2018",
    address = "Brussels, Belgium"
}

@InProceedings{yu2019sparc,
  author =      "Tao Yu, Rui Zhang, Michihiro Yasunaga, Yi Chern Tan, Xi Victoria Lin, Suyi Li, Heyang Er, Irene Li, Bo Pang, Tao Chen, Emily Ji, Shreya Dixit, David Proctor, Sungrok Shim, Jonathan Kraft, Vincent Zhang, Caiming Xiong, Richard Socher, Dragomir Radev",
  title =       "SParC: Cross-Domain Semantic Parsing in Context",
  booktitle =   "Proceedings of The 57th Annual Meeting of the Association for Computational Linguistics",
  year =        "2019",
  address =     "Florence, Italy"
}

@InProceedings{yu2019cosql,
  author =      "Tao Yu, Rui Zhang, He Yang Er, Suyi Li, Eric Xue, Bo Pang, Xi Victoria Lin, Yi Chern Tan, Tianze Shi, Zihan Li, Youxuan Jiang, Michihiro Yasunaga, Sungrok Shim, Tao Chen, Alexander Fabbri, Zifan Li, Luyao Chen, Yuwen Zhang, Shreya Dixit, Vincent Zhang, Caiming Xiong, Richard Socher, Walter Lasecki, Dragomir Radev",
  title =       "CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases",
  booktitle =   "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  year =        "2019",
  address =     "Hong Kong, China"
}

@InProceedings{zhang2019editing,
  author =      "Rui Zhang, Tao Yu, He Yang Er, Sungrok Shim, Eric Xue, Xi Victoria Lin, Tianze Shi, Caiming Xiong, Richard Socher, Dragomir Radev",
  title =       "Editing-Based SQL Query Generation for Cross-Domain Context-Dependent Questions",
  booktitle =   "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  year =        "2019",
  address =     "Hong Kong, China"
}
```

Contact Rui Zhang for any question.

### Dependency

The model is tested in python 3.6 and pytorch 1.0. We recommend using `conda` and `pip`:

```
conda create -n editsql python=3.6
source activate editsql
pip install -r requirements.txt
```

The evaluation scripts use python 2.7

Download Pretrained BERT model from [here](https://drive.google.com/file/d/1f_LEWVgrtZLRuoiExJa5fNzTS8-WcAX9/view?usp=sharing) as `model/bert/data/annotated_wikisql_and_PyTorch_bert_param/pytorch_model_uncased_L-12_H-768_A-12.bin`.

Download the database sqlite files from [here](https://drive.google.com/file/d/1a828mkHcgyQCBgVla0jGxKJ58aV8RsYK/view?usp=sharing) as `data/database`.

### Run Spider experiment
First, download [Spider](https://yale-lily.github.io/spider). Then please follow

- `run_spider_editsql.sh`. We saved our experimental logs at `logs/logs_spider_editsql`. The dev results can be reproduced by `test_spider_editsql.sh` with the trained model downloaded from [here](https://drive.google.com/file/d/1KwXIdJBYKG0-PzCi1GvvSnUxJzxNq_CL/view?usp=sharing) and put under `logs/logs_spider_editsql/save_12`.

This reproduces the Spider result in "Editing-Based SQL Query Generation for Cross-Domain Context-Dependent Questions".

<table>
  <tr>
    <td></td>
    <td>Dev</td>
    <td>Test</td>
  </tr>
  <tr>
    <td>EditSQL</td>
    <td>57.6</td>
    <td>53.4</td>
  </tr>
</table>

### Run SParC experiment

First, download [SParC](https://yale-lily.github.io/sparc). Then please follow

- use cdseq2seq: `run_sparc_cdseq2seq.sh`. We saved our experimental logs at `logs/logs_sparc_cdseq2seq`
- use cdseq2seq with segment copy:  `run_sparc_cdseq2seq_segment_copy.sh`. We saved our experimental logs at `logs/logs_sparc_cdseq2seq_segment_copy`
- use editsql: `run_sparc_editsql.sh`. We saved our experimental logs at `logs/logs_sparc_editsql`. The dev results can be reproduced by `test_sparc_editsql.sh` with the trained model downloaded from [here](https://drive.google.com/file/d/1MRN3_mklw8biUphFxmD7OXJ57yS-FkJP/view?usp=sharing) and put under `logs/logs_sparc_editsql/save_31_sparc_editsql`.

This reproduces the SParC result in "Editing-Based SQL Query Generation for Cross-Domain Context-Dependent Questions".

<table>
  <tr>
    <th></th>
    <th colspan="2">Question Match</th>
    <th colspan="2">Interaction Match</th>
  </tr>
  <tr>
    <td></td>
    <td>Dev</td>
    <td>Test</td>
    <td>Dev</td>
    <td>Test</td>
  </tr>
  <tr>
    <td>CD-Seq2Seq</td>
    <td>21.9</td>
    <td>-</td>
    <td>8.1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>CD-Seq2Seq+segment copy (use predicted query)</td>
    <td>21.7</td>
    <td>-</td>
    <td>9.5</td>
    <td>-</td>
  </tr>
  <tr>
    <td>CD-Seq2Seq+segment copy (use gold query)</td>
    <td>27.3</td>
    <td>-</td>
    <td>10.0</td>
    <td>-</td>
  </tr>
  <tr>
    <td>EditSQL (use predicted query)</td>
    <td>47.2</td>
    <td>47.9</td>
    <td>29.5</td>
    <td>25.3</td>
  </tr>
  <tr>
    <td>EditSQL (use gold query)</td>
    <td>53.4</td>
    <td>54.5</td>
    <td>29.2</td>
    <td>25.0</td>
  </tr>
</table>

### Run CoSQL experiment

First, download [CoSQL](https://yale-lily.github.io/cosql). Then please follow

- `run_cosql_cdseq2seq.sh`. We saved our experimental logs at `logs/logs_cosql_cdseq2seq`.
- `run_cosql_editsql.sh`. We saved our experimental logs at `logs/logs_cosql_editsql`. The dev results can be reproduced by `test_cosql_editsql.sh` with the trained model downloaded from [here](https://drive.google.com/file/d/1ggf05rLVUpqamkEFbhu2CX35-PTGpFx4/view?usp=sharing) and put under `logs/logs_cosql_editsql/save_12_cosql_editsql`.

This reproduces the SQL-grounded dialog state tracking result in "CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases".

<table>
  <tr>
    <th></th>
    <th colspan="2">Question Match</th>
    <th colspan="2">Interaction Match</th>
  </tr>
  <tr>
    <td></td>
    <td>Dev</td>
    <td>Test</td>
    <td>Dev</td>
    <td>Test</td>
  </tr>
  <tr>
    <td>CD-Seq2Seq</td>
    <td>13.8</td>
    <td>13.9</td>
    <td>2.1</td>
    <td>2.6</td>
  </tr>
  <tr>
    <td>EditSQL</td>
    <td>39.9</td>
    <td>40.8</td>
    <td>12.3</td>
    <td>13.7</td>
  </tr>
</table>

### Run ATIS experiment

To get ATIS data and get evaluation on the result accuracy, you need get ATIS data from [here](https://github.com/lil-lab/atis), set up your mysql database for ATIS and change `--database_username` and `--database_password` in `parse_args.py`.

Please follow `run_atis.sh`

This reproduces the ATIS result in "Learning to map context dependent sentences to executable formal queries".
We saved our experimental logs at `logs/logs_atis`

<table>
  <tr>
    <th></th>
    <th colspan="3">Dev</th>
    <th colspan="3">Test</th>
  </tr>
  <tr>
    <td></td>
    <td>Query</td>
    <td>Relaxed</td>
    <td>Strict</td>
    <td>Query</td>
    <td>Relaxed</td>
    <td>Strict</td>
  </tr>
  <tr>
    <td>Suhr et al., 2018</td>
    <td>37.5(0.9)</td>
    <td>63.0(0.7)</td>
    <td>62.5(0.9)</td>
    <td>43.6(1.0)</td>
    <td>69.3(0.8)</td>
    <td>69.2(0.8)</td>
  </tr>
  <tr>
    <td>Our Replication</td>
    <td>38.8</td>
    <td>63.3</td>
    <td>62.8</td>
    <td>44.6</td>
    <td>68.3</td>
    <td>68.2</td>
  </tr>
</table>

### Acknowledgement

This implementation is based on "[Learning to map context dependent sentences to executable formal queries](https://github.com/lil-lab/atis)". Alane Suhr, Srinivasan Iyer, and Yoav Artzi. In NAACL, 2018.
