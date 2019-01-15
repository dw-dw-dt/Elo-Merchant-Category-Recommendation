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
from utils import reduce_mem_usage, one_hot_encoder
from features_base import Feature, get_arguments, generate_features


# featureの格納場所はfeatures
Feature.dir = 'features'


def load_df():
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
    df = reduce_mem_usage(df)
    return df


def load_merchants():
    merchants_df = feather.read_dataframe('./data/input/merchants.feather')
    merchants_df = reduce_mem_usage(merchants_df)
    return merchants_df


class Traintest_simple(Feature):
    def create_features(self):
        df = load_df()
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


class Traintest_complex(Feature):
    def create_features(self):
        df = load_df()
        _df = df.set_index('card_id')
        new_df, new_cols = one_hot_encoder(_df, nan_as_category=False)
        for f in ['feature_1','feature_2','feature_3']:
            order_label = new_df.groupby([f])['outliers'].mean()
            new_df[f] = new_df[f].map(order_label)
        #print(new_df.head(10))
        new_df = new_df.reset_index()
        #print(new_df.head(10))
        for col in new_df.columns:
            if col not in _df.columns and col != 'card_id':
                self.df[col] = new_df[col].astype(int)
        self.df['feature_sum'] = new_df['feature_1'].astype(int) + new_df['feature_2'].astype(int) + new_df['feature_3'].astype(int)
        self.df['feature_mean'] = self.df['feature_sum'].astype(float)/3
        self.df['feature_max'] = new_df[['feature_1', 'feature_2', 'feature_3']].max(axis=1).astype(int)
        self.df['feature_min'] = new_df[['feature_1', 'feature_2', 'feature_3']].min(axis=1).astype(int)
        self.df['feature_var'] = new_df[['feature_1', 'feature_2', 'feature_3']].std(axis=1).astype(int)
        self.df['feature_skew'] = new_df[['feature_1', 'feature_2', 'feature_3']].skew(axis=1).astype(int)


class Marchants_simple(Feature):
    def create_features(self):
        merchants_df = load_merchants()
        self.df['category_1'] = merchants_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
        self.df['category_4'] = merchants_df['category_4'].map({'Y': 1, 'N': 0}).astype(int)
        self.df['avg_numerical'] = merchants_df[['numerical_1','numerical_2']].mean(axis=1).astype(float) # 型でエラーが出たのでmapでfloatに変換
        self.df['avg_sales'] = merchants_df[['avg_sales_lag3','avg_sales_lag6','avg_sales_lag12']].mean(axis=1).astype(float)
        self.df['avg_purchases'] = merchants_df[['avg_purchases_lag3','avg_purchases_lag6','avg_purchases_lag12']].mean(axis=1).astype(float)
        self.df['avg_active_months'] = merchants_df[['active_months_lag3','active_months_lag6','active_months_lag12']].mean(axis=1).astype(float)
        self.df['max_sales'] = merchants_df[['avg_sales_lag3','avg_sales_lag6','avg_sales_lag12']].max(axis=1).astype(float)
        self.df['max_purchases'] = merchants_df[['avg_purchases_lag3','avg_purchases_lag6','avg_purchases_lag12']].max(axis=1).astype(float)
        self.df['max_active_months'] = merchants_df[['active_months_lag3','active_months_lag6','active_months_lag12']].max(axis=1).astype(int)
        self.df['min_sales'] = merchants_df[['avg_sales_lag3','avg_sales_lag6','avg_sales_lag12']].min(axis=1).astype(float)
        self.df['min_purchases'] = merchants_df[['avg_purchases_lag3','avg_purchases_lag6','avg_purchases_lag12']].min(axis=1).astype(float)
        self.df['min_active_months'] = merchants_df[['active_months_lag3','active_months_lag6','active_months_lag12']].min(axis=1).astype(int)
        self.df['sum_category'] = merchants_df[['category_1','category_2','category_4']].sum(axis=1).astype(float)
        # fillna
        self.df['category_2'] = merchants_df['category_2'].fillna(-1).astype(int).astype(object)

"""
class Marchants_complex(Feature):
    def create_features(self):
        new_merchants_df, new_cols = one_hot_encoder(merchants_df, nan_as_category=False)
        # unique columns
        col_unique =['merchant_group_id', 'merchant_category_id', 'subsector_id',
                    'city_id', 'state_id']

        # aggregation
        aggs = {}
        for col in col_unique:
            aggs[col] = ['nunique']

        aggs['numerical_1'] = ['mean','max','min','std','var']
        aggs['numerical_2'] = ['mean','max','min','std','var']
        aggs['avg_sales_lag3'] = ['mean','max','min','std','var']
        aggs['avg_sales_lag6'] = ['mean','max','min','std','var']
        aggs['avg_sales_lag12'] = ['mean','max','min','std','var']
        aggs['avg_purchases_lag3'] = ['mean','max','min','std','var']
        aggs['avg_purchases_lag6'] = ['mean','max','min','std','var']
        aggs['avg_purchases_lag12'] = ['mean','max','min','std','var']
        aggs['active_months_lag3'] = ['mean','max','min','std','var']
        aggs['active_months_lag6'] = ['mean','max','min','std','var']
        aggs['active_months_lag12'] = ['mean','max','min','std','var']
        aggs['category_1'] = ['mean']
        aggs['category_4'] = ['mean']
        aggs['most_recent_sales_range_A'] = ['mean']
        aggs['most_recent_sales_range_B'] = ['mean']
        aggs['most_recent_sales_range_C'] = ['mean']
        aggs['most_recent_sales_range_D'] = ['mean']
        aggs['most_recent_sales_range_E'] = ['mean']
        aggs['most_recent_purchases_range_A'] = ['mean']
        aggs['most_recent_purchases_range_B'] = ['mean']
        aggs['most_recent_purchases_range_C'] = ['mean']
        aggs['most_recent_purchases_range_D'] = ['mean']
        aggs['most_recent_purchases_range_E'] = ['mean']
        aggs['category_2_-1'] = ['mean']
        aggs['category_2_1'] = ['mean']
        aggs['category_2_2'] = ['mean']
        aggs['category_2_3'] = ['mean']
        aggs['category_2_4'] = ['mean']
        aggs['category_2_5'] = ['mean']
        aggs['avg_numerical'] = ['mean','max','min','std','var']
        aggs['avg_sales'] = ['mean','max','min','std','var']
        aggs['avg_purchases'] = ['mean','max','min','std','var']
        aggs['avg_active_months'] = ['mean','max','min','std','var']
        aggs['max_sales'] = ['mean','max','min','std','var']
        aggs['max_purchases'] = ['mean','max','min','std','var']
        aggs['max_active_months'] = ['mean','max','min','std','var']
        aggs['min_sales'] = ['mean','max','min','std','var']
        aggs['min_purchases'] = ['mean','max','min','std','var']
        aggs['min_active_months'] = ['mean','max','min','std','var']
        aggs['sum_category'] = ['mean']

        new_merchants_df = new_merchants_df.reset_index().groupby('merchant_id').agg(aggs)

        # カラム名の変更
        new_merchants_df.columns = pd.Index([e[0] + "_" + e[1] for e in new_merchants_df.columns.tolist()])
        new_merchants_df.columns = ['mer_'+ c for c in new_merchants_df.columns]
        for col in new_merchants_df.columns:
            self.df[col] = new_merchants_df[col]
"""

if __name__ == '__main__':
    args = get_arguments()
    generate_features(globals(), args.force)
