import pandas as pd
import datetime
import logging
from sklearn.model_selection import KFold
#import argparse
import gc
import feather
import json
import subprocess
import os
import sys
this_folder = '/user01'
cwd = os.getcwd()
sys.path.append(cwd.replace(this_folder,''))
#from lgbmClassifier import train_and_predict
from kfold_lgbm import kfold_lightgbm
from utils import log_best, load_datasets, display_importances, save2pkl, line_notify, submit #, load_target

# config
create_features = False # create_features.py を再実行する場合は True, そうでない場合は False
is_debug = True # True だと少数のデータで動かします, False だと全データを使います. また NUM_FOLDS = 2 になります
use_GPU = False
COMPETITION_NAME = 'elo-merchant-category-recommendation'
FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'Outlier_Likelyhood', 'OOF_PRED', 'outliers_pred', 'month_0']
NUM_FOLDS = 11 # is_debug が True だとここの設定によらず 2 に更新されます
LOSS = 'rmse'


# start log
now = datetime.datetime.now()
logging.basicConfig(
    filename='../logs/log_{0:%Y-%m-%d-%H-%M-%S}.log'.format(now), level=logging.DEBUG
)
logging.debug('../logs/log_{0:%Y-%m-%d-%H-%M-%S}.log'.format(now))

# create features
if create_features == True:
    result = subprocess.run('python create_features.py', shell=True)
    if result.returncode != 0:
        print('ERROR')
        quit()

# loading
path = cwd.replace(this_folder,'/features')
train_df, test_df = load_datasets(path)

# debug or not
if is_debug == True:
    train_df = train_df.iloc[0:10000]
    test_df = test_df.iloc[0:10000]
    NUM_FOLDS = 2
logging.debug("Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

# model
models, model_params, feature_importance_df, test_preds = kfold_lightgbm(train_df,
                                                                         test_df,
                                                                         model_loss=LOSS,
                                                                         num_folds=NUM_FOLDS,
                                                                         feats_exclude=FEATS_EXCLUDED,
                                                                         stratified=False,
                                                                         use_gpu=use_GPU)
# CVスコア
scores = [
    m.best_score['valid'][LOSS] for m in models
]
score = sum(scores) / len(scores)
#display_importances(feature_importance_df,
#                    '../models/lgbm_importances.png',
#                    '../models/feature_importance_lgbm.csv')

print('===CV scores===')
print(scores)
print(score)
logging.debug('===CV scores===')
logging.debug(scores)
logging.debug(score)

print('finish')
quit()











################
y_preds = []
models = []

lgbm_params = {
    "learning_rate": 0.1,
    "num_leaves": 8,
    "boosting_type": "gbdt",
    "colsample_bytree": 0.65,
    "reg_alpha": 1,
    "reg_lambda": 1,
    "objective": "multiclass",
    "num_class": 2
}

kf = KFold(n_splits=3, random_state=0)
for n_fold, train_index, valid_index in enumerate(kf.split(train_df[feats]), train_df['outliers']):
    X_train, X_valid = (
        X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :]
    )
    y_train, y_valid = y_train_all[train_index], y_train_all[valid_index]

    # lgbmの実行
    y_pred, model = train_and_predict(
        X_train, X_valid, y_train, y_valid, X_test, lgbm_params
    )

    # 結果の保存
    y_preds.append(y_pred)
    models.append(model)

    # スコア
    log_best(model, config['loss'])

# CVスコア
scores = [
    m.best_score['valid_0'][config['loss']] for m in models
]
score = sum(scores) / len(scores)

# モデルの保存
for i, model in enumerate(models):
    save2pkl('../models/lgbm_{0}_{1}.pkl'.format(score, i), model)

# モデルパラメータの保存
with open('../models/lgbm_{0}_params.json'.format(score), 'w') as f:
    json.dump(lgbm_params, f, indent=4)

print('===CV scores===')
print(scores)
print(score)
logging.debug('===CV scores===')
logging.debug(scores)
logging.debug(score)

# submitファイルの作成
ID_name = config['ID_name']
sub = pd.DataFrame(pd.read_csv('../data/input/test.csv')[ID_name])

for i in range(len(y_preds) - 1):
    y_preds[0] += y_preds[i + 1]

sub[target_name] = [1 if y > 1 else 0 for y in y_preds[0]]

sub.to_csv(
    '../data/output/submit_{0:%Y-%m-%d-%H-%M-%S}_{1}.csv'.format(now, score),
    index=False
)
