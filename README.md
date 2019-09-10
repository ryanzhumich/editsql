# CD-Seq2Seq for SParC and CoSQL

This is a pytorch implementation of the CD-Seq2Seq baseline in the following papers
- "[SParC: Cross-Domain Semantic Parsing in Context](https://arxiv.org/abs/1906.02285)", ACL 2019
- "[CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases](http://croma.eecs.umich.edu/pubs/coSQL_EMNLP2019.pdf)", EMNLP 2019
- "[Editing-Based SQL Query Generation for Cross-Domain Context-Dependent Questions](https://arxiv.org/pdf/1909.00786.pdf)", EMNLP 2019

Please cite the papers if you use our data and code.
```
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
conda create -n cdseq2seq python=3.6
source activate cdseq2seq
pip install -r requirements.txt
```

The evaluation scripts use python 2.7

### Run SParC experiment

First, download SParC from [here](https://yale-lily.github.io/sparc). Then please follow

- `run_sparc_cdseq2seq.sh`. We saved our experimental logs at `logs/logs_sparc_cdseq2seq`
- use segment copy:  `run_sparc_cdseq2seq_segment_copy.sh`. We saved our experimental logs at `logs/logs_sparc_cdseq2seq_segment_copy`

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
</table>

### Run CoSQL experiment

First, download CoSQL from [here](https://yale-lily.github.io/cosql). Then please follow

- `run_cosql_cdseq2seq.sh`. We saved our experimental logs at `logs/logs_cosql_cdseq2seq`

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
