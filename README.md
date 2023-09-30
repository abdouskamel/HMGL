# Link Prediction on High-Dimensional Multiplex Graphs

## Requirements
Python 3.6 <br />
numpy <br />
scipy <br />
scikit-learn <br />
pytorch <br />
tqdm

## Run
`main.py DATASET_NAME`

- Specify the dataset name in `DATASET_NAME`. The possible values are the following : biogrid_4503_bis, biogrid_4211, dblp_5124, imdb_3000, STRING-DB_4083

Examples :

`main.py biogrid_4211` <br />
`main.py dblp_5124`

## Reference
If you find this work useful in your research, please consider citing the following extended paper:

```
@ARTICLE{abdous2023hierarchical,
  author={Abdous, Kamel and Mrabah, Nairouz and Bouguessa, Mohamed},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Hierarchical Aggregations for High-Dimensional Multiplex Graph Embedding}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TKDE.2023.3305809}
}
```
