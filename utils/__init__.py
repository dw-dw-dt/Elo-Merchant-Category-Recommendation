import time
import pandas as pd
import numpy as np
import feather
from contextlib import contextmanager
import pickle


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_datasets(feats):
    #dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    dfs = [feather.read_dataframe(f'features/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    #dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    dfs = [feather.read_dataframe(f'features/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv('./data/input/train.csv')
    y_train = train[target_name]
    return y_train


def save2pkl(path, df):
    f = open(path, 'wb')
    pickle.dump(df, f)
    f.close


def load_pkl(path):
    f = open(path, 'rb')
    out = pickle.load(f)
    return out


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    int8_min = np.iinfo(np.int8).min
    int8_max = np.iinfo(np.int8).max
    int16_min = np.iinfo(np.int16).min
    int16_max = np.iinfo(np.int16).max
    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max
    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max
    float16_min = np.finfo(np.float16).min
    float16_max = np.finfo(np.float16).max
    float32_min = np.finfo(np.float32).min
    float32_max = np.finfo(np.float32).max
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > int8_min and c_max < int8_max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > int16_min and c_max < int16_max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > int32_min and c_max < int32_max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > int64_min and c_max < int64_max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > float16_min and c_max < float16_max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > float32_min and c_max < float32_max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    mem_diff_pct = (start_mem - end_mem) / start_mem
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * mem_diff_pct))
    return df
