# CD-Seq2Seq for SParC

This is a pytorch implementation of the CD-Seq2Seq baseline in our ACL 2019 paper "[SParC: Cross-Domain Semantic Parsing in Context](https://arxiv.org/abs/1906.02285)".

Please cite this paper if you use our data/code.
```
@InProceedings{Yu2019,
  author =      "Tao Yu, Rui Zhang, Michihiro Yasunaga, Yi Chern Tan, Xi Victoria Lin, Suyi Li, Heyang Er, Irene Li, Bo Pang, Tao Chen, Emily Ji, Shreya Dixit, David Proctor, Sungrok Shim, Jonathan Kraft, Vincent Zhang, Caiming Xiong, Richard Socher, Dragomir Radev",
  title =       "SParC: Cross-Domain Semantic Parsing in Context",
  booktitle =   "Proceedings of The 57th Annual Meeting of the Association for Computational Linguistics",
  year =        "2019",
  address =     "Florence, Italy"
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

Please follow

- `run_sparc_cdseq2seq.sh`. We saved our experimental logs at `logs/logs_sparc_cdseq2seq`
- use segment copy:  `run_sparc_cdseq2seq_segment_copy.sh`. We saved our experimental logs at `logs/logs_sparc_cdseq2seq_segment_copy`

This reproduces the SParC result in "SParC: Cross-Domain Semantic Parsing in Context"

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
