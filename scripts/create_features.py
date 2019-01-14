import pandas as pd
import numpy as np
import datetime
import gc
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd.replace('/scripts',''))
from utils import load_pkl, save2pkl, one_hot_encoder


def train_test():
    train_df = load_pkl('./data/input/train.pkl')
    test_df = load_pkl('./data/input/test.pkl')
    train_df.loc[:,'is_test'] = 0
    test_df.loc[:,'is_test'] = 1

    df = train_df.append(test_df)
    # 処理を書く

    return df

if __name__ == '__main__':
    df = train_test()
    train = df[df.is_test == 0]
    test = df[df.is_test == 1]
    for col in df.columns:
        save2pkl('./features/train_{}.pkl'.format(col), train.loc[:,col])
        save2pkl('./features/test_{}.pkl'.format(col), test.loc[:,col])
