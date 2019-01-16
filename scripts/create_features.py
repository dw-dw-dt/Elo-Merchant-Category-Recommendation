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


class Train_test(Feature):
    def create_features(self):
        # load csv
        #train_df = pd.read_csv('../input/train.csv', index_col=['card_id'], nrows=num_rows)
        train_df = feather.read_dataframe('./data/input/train.feather')
        train_df = train_df.set_index('card_id')
        #test_df = pd.read_csv('../input/test.csv', index_col=['card_id'], nrows=num_rows)
        test_df = feather.read_dataframe('./data/input/test.feather')
        test_df = test_df.set_index('card_id')

        print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

        # outlier
        train_df['outliers'] = 0
        train_df.loc[train_df['target'] < -30, 'outliers'] = 1

        # set target as nan
        test_df['target'] = np.nan

        # merge
        df = train_df.append(test_df)

        del train_df, test_df
        gc.collect()

        # datetimeへ変換
        df['first_active_month'] = pd.to_datetime(df['first_active_month'])

        # datetime features
    #    df['month'] = df['first_active_month'].dt.month.fillna(0).astype(int).astype(object)
    #    df['year'] = df['first_active_month'].dt.year.fillna(0).astype(int).astype(object)
    #    df['dayofweek'] = df['first_active_month'].dt.dayofweek.fillna(0).astype(int).astype(object)
    #    df['weekofyear'] = df['first_active_month'].dt.weekofyear.fillna(0).astype(int).astype(object)
        df['quarter'] = df['first_active_month'].dt.quarter
    #    df['month_year'] = df['month'].astype(str)+'_'+df['year'].astype(str)
        df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

        df['days_feature1'] = df['elapsed_time'] * df['feature_1']
        df['days_feature2'] = df['elapsed_time'] * df['feature_2']
        df['days_feature3'] = df['elapsed_time'] * df['feature_3']

        df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
        df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
        df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']

        # one hot encoding
        df, cols = one_hot_encoder(df, nan_as_category=False)

        for f in ['feature_1','feature_2','feature_3']:
            order_label = df.groupby([f])['outliers'].mean()
            df[f] = df[f].map(order_label)

        df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
        df['feature_mean'] = df['feature_sum']/3
        df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
        df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
        df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)
        df['feature_skew'] = df[['feature_1', 'feature_2', 'feature_3']].skew(axis=1)

        df = df.reset_index()
        self.df = df


class Historical_transaction(Feature):
    def create_features(self):
        #hist_df = pd.read_csv('../input/historical_transactions.csv', nrows=num_rows)
        hist_df = feather.read_dataframe('./data/input/historical_transactions.feather')

        # fillna
        hist_df['category_2'].fillna(1.0,inplace=True)
        hist_df['category_3'].fillna('A',inplace=True)
        hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
        hist_df['installments'].replace(-1, np.nan,inplace=True)
        hist_df['installments'].replace(999, np.nan,inplace=True)

        # Y/Nのカラムを1-0へ変換
        hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
        hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
        hist_df['category_3'] = hist_df['category_3'].map({'A':0, 'B':1, 'C':2})

        # datetime features
        hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    #    hist_df['year'] = hist_df['purchase_date'].dt.year
        hist_df['month'] = hist_df['purchase_date'].dt.month
        hist_df['day'] = hist_df['purchase_date'].dt.day
        hist_df['hour'] = hist_df['purchase_date'].dt.hour
        hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
        hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
        hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >=5).astype(int)

        # additional features
        hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']

        #ブラジルの休日
        cal = Brazil()
        hist_df['is_holiday'] = hist_df['purchase_date'].dt.date.apply(cal.is_holiday).astype(int)

        # 購入日からイベント日までの経過日数
        #Christmas : December 25 2017
        hist_df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Mothers Day: May 14 2017
        hist_df['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #fathers day: August 13 2017
        hist_df['fathers_day_2017']=(pd.to_datetime('2017-08-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Childrens day: October 12 2017
        hist_df['Children_day_2017']=(pd.to_datetime('2017-10-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Valentine's Day : 12th June, 2017
        hist_df['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
        #Black Friday : 24th November 2017
        hist_df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

        #2018
        #Mothers Day: May 13 2018
        hist_df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

        hist_df['month_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days)//30
        hist_df['month_diff'] += hist_df['month_lag']

        # additional features
        hist_df['duration'] = hist_df['purchase_amount']*hist_df['month_diff']
        hist_df['amount_month_ratio'] = hist_df['purchase_amount']/hist_df['month_diff']

        # memory usage削減
        hist_df = reduce_mem_usage(hist_df)

        col_unique =['subsector_id', 'merchant_id', 'merchant_category_id']
        col_seas = ['month', 'hour', 'weekofyear', 'day']

        aggs = {}
        for col in col_unique:
            aggs[col] = ['nunique']

        for col in col_seas:
            aggs[col] = ['nunique', 'mean', 'min', 'max']

        aggs['purchase_amount'] = ['sum','max','min','mean','var','skew']
        aggs['installments'] = ['sum','max','mean','var','skew']
        aggs['purchase_date'] = ['max','min']
        aggs['month_lag'] = ['max','min','mean','var','skew']
        aggs['month_diff'] = ['max','min','mean','var','skew']
        aggs['authorized_flag'] = ['mean']
        aggs['weekend'] = ['mean', 'max']
        aggs['weekday'] = ['nunique', 'mean'] # overwrite
        aggs['category_1'] = ['mean']
        aggs['category_2'] = ['mean']
        aggs['category_3'] = ['mean']
        aggs['card_id'] = ['size','count']
        aggs['is_holiday'] = ['mean']
        aggs['price'] = ['sum','mean','max','min','var','skew']
        aggs['Christmas_Day_2017'] = ['mean']
        aggs['Mothers_Day_2017'] = ['mean']
        aggs['fathers_day_2017'] = ['mean']
        aggs['Children_day_2017'] = ['mean']
        aggs['Valentine_Day_2017'] = ['mean']
        aggs['Black_Friday_2017'] = ['mean']
        aggs['Mothers_Day_2018'] = ['mean']
        aggs['duration']=['mean','min','max','var','skew']
        aggs['amount_month_ratio']=['mean','min','max','var','skew']

        for col in ['category_2','category_3']:
            hist_df[col+'_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
            hist_df[col+'_min'] = hist_df.groupby([col])['purchase_amount'].transform('min')
            hist_df[col+'_max'] = hist_df.groupby([col])['purchase_amount'].transform('max')
            hist_df[col+'_sum'] = hist_df.groupby([col])['purchase_amount'].transform('sum')
            aggs[col+'_mean'] = ['mean']

        hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

        # カラム名の変更
        hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
        hist_df.columns = ['hist_'+ c for c in hist_df.columns]

        hist_df['hist_purchase_date_diff'] = (hist_df['hist_purchase_date_max']-hist_df['hist_purchase_date_min']).dt.days
        hist_df['hist_purchase_date_average'] = hist_df['hist_purchase_date_diff']/hist_df['hist_card_id_size']
        hist_df['hist_purchase_date_uptonow'] = (datetime.datetime.today()-hist_df['hist_purchase_date_max']).dt.days
        hist_df['hist_purchase_date_uptomin'] = (datetime.datetime.today()-hist_df['hist_purchase_date_min']).dt.days

        hist_df = hist_df.reset_index()

        self.df = hist_df


if __name__ == '__main__':
    args = get_arguments()
    generate_features(globals(), args.force)
