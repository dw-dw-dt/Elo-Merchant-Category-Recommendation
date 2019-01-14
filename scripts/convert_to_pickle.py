import pandas as pd
import gc
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd.replace('/scripts',''))
from utils import save2pkl, reduce_mem_usage

################
## data/input の train.csv, test.csv をtrain.pkl, test.pkl に変換するスクリプト
################

target = [
    'train',
    'test'
]

extension = 'csv'
# extension = 'tsv'
# extension = 'zip'

for t in target:
    _temp = pd.read_csv('./data/input/' + t + '.' + extension, encoding="utf-8")
    _temp = reduce_mem_usage(_temp)
    save2pkl('./data/input/' + t + '.pkl', _temp)
    del _temp
    gc.collect()
