import pandas as pd
import numpy as np
import datetime
import feather
import gc
from workalendar.america import Brazil
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd.replace('/scripts',''))
from utils import one_hot_encoder
from features_base import Feature, get_arguments, generate_features


# featureの格納場所はfeatures
Feature.dir = 'features'
# train, test はグローバルに宣言
train = feather.read_dataframe('./data/input/train.feather')
test = feather.read_dataframe('./data/input/train.feather')
train.loc[:,'is_test'] = 0
test.loc[:,'is_test'] = 1
train['outliers'] = 0
train.loc[train['target'] < -30, 'outliers'] = 1
test['target'] = np.nan
df = train.append(test)
del train, test
gc.collect()
df = df.reset_index()


class Activemonthclass(Feature):
    def create_features(self):
        self.df['first_active_month'] = pd.to_datetime(df['first_active_month'])
        #self.df['month'] = self.df['first_active_month'].dt.month.fillna(0).astype(int).astype(object)
        #self.df['year'] = self.df['first_active_month'].dt.year.fillna(0).astype(int).astype(object)
        #self.df['dayofweek'] = self.df['first_active_month'].dt.dayofweek.fillna(0).astype(int).astype(object)
        #self.df['weekofyear'] = self.df['first_active_month'].dt.weekofyear.fillna(0).astype(int).astype(object)
        self.df['quarter'] = self.df['first_active_month'].dt.quarter
        self.df['elapsed_time'] = (datetime.datetime.today() - self.df['first_active_month']).dt.days
        self.df['days_feature1'] = self.df['elapsed_time'] * df['feature_1']
        self.df['days_feature2'] = self.df['elapsed_time'] * df['feature_2']
        self.df['days_feature3'] = self.df['elapsed_time'] * df['feature_3']
        self.df['days_feature1_ratio'] = df['feature_1'] / self.df['elapsed_time']
        self.df['days_feature2_ratio'] = df['feature_2'] / self.df['elapsed_time']
        self.df['days_feature3_ratio'] = df['feature_3'] / self.df['elapsed_time']
        df, cols = one_hot_encoder(df, nan_as_category=False)
        for f in ['feature_1','feature_2','feature_3']:
            order_label = df.groupby([f])['outliers'].mean()
            df[f] = df[f].map(order_label)
        self.df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
        self.df['feature_mean'] = self.df['feature_sum']/3
        self.df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
        self.df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
        self.df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)
        self.df['feature_skew'] = df[['feature_1', 'feature_2', 'feature_3']].skew(axis=1)


if __name__ == '__main__':
    args = get_arguments()
    generate_features(globals(), args.force)

