import pandas as pd
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd.replace('/scripts',''))
from utils import save2pkl

################
## data/input の train.csv, test.csv をtrain.pkl, test.pkl に変換するスクリプト
################

target = [
    'train',
    'test',
]

extension = 'csv'
# extension = 'tsv'
# extension = 'zip'

for t in target:
    _temp = pd.read_csv('./data/input/' + t + '.' + extension, encoding="utf-8")
    save2pkl('./data/input/' + t + '.pkl', _temp)
