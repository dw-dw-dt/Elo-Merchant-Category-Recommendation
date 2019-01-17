# フォルダ構造は以下の通り.
```
.
├── README.md
├── configs
│   └── default.json
├── convert_to_feather.py
├── create_features.py
├── data
│   ├── input
│   │   ├── *.csv
│   │   └── *.feather
│   └── output
│
├── features:特徴量のfeatherファイルの置き場
│   └── *.feather
│
├── logs
│   └── logger.py
│
├── models
│   └── lgbmClassifier.py
│
├── notebook
│   └── *.ipynb
├── run.py
│
├── src:複雑なクラスなどを定義したpyファイルの置き場
│   └── features_base.py
│
└── utils
    └── __init__.py:便利関数の置き場
```

# 使い方
初めは, 以下で train.csv, test.csv を feather に変換する.(基本的に一度きり)
```
python convert_to_feather.py
```
そして, 特徴量の feather化 を以下で行う.(特徴量に変更があった時は, 該当する特徴量ファイル（/features/*.feather）を削除の上で, 再実行.あるいは別のクラスを定義して実行)
```
python create_features.py
```
通常は実行するのは, 以下のみ.
```
python run.py
```
