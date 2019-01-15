import pandas as pd
import numpy as np
import datetime
import feather
import gc
from workalendar.america import Brazil
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


if __name__ == '__main__':
    args = get_arguments()
    generate_features(globals(), args.force)

