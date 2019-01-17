import pandas as pd


target = [
    'train',
    'test',
    'historical_transactions',
    'merchants',
    'new_merchant_transactions'
]

extension = 'csv'
# extension = 'tsv'
# extension = 'zip'

for t in target:
    (pd.read_csv('./data/input/' + t + '.' + extension, encoding="utf-8"))\
     .to_feather('./data/input/' + t + '.feather')
