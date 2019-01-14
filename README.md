For Kaggle use

フォルダ構造は以下の通り.
```
.
├── README.md
├── configs
│   └── default.json
├── data
│   ├── input
│   │   ├── sample_submission.csv
│   │   ├── test.csv
│   │   └── train.csv
│   └── output
├── features
│   ├── __init__.py
│   ├── base.py
│   └── create.py
├── logs
│   ├── log_20190114194353.log
│   └── logger.py
├── models
│   └── lgbmClassifier.py
├── notebook
│   └── eda.ipynb
├── run.py
├── scripts
│   └── convert_to_feather.py
└── utils
    └── __init__.py
```

初めは, 以下で train.csv, test.csv をpklに変換する.
```
python scripts/convert_to_pickle.py
```
そして, 特徴量の feather化を以下で行う.
```
python scripts/create_features.py
```
