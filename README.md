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
├── features
│   └── *.feather
├── logs
│   └── logger.py
├── models
│   └── lgbmClassifier.py
├── notebook
├── run.py
├── src
│   └── features_base.py
└── utils
    └── __init__.py
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
