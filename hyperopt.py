# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:51:29 2020

@author: Idrissa
"""

import gc 
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, KFold, TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING, partial

# Define searched space
hyper_space = {'metric':'auc','boosting':'gbdt','min_data_in_leaf': 99,'n_estimators':300,
               'objective': hp.choice('objective', ['regression_l1','regression_l2','huber']), 
               'learning_rate': hp.choice('learning_rate', [0.05, .1, .3]),
               'max_depth': hp.choice('max_depth',np.arange(6, 25, dtype=int)),
               'num_leaves': hp.choice('num_leaves', np.arange(16, 1024, 8, dtype=int)),
               'max_bin':hp.choice('max_bin',np.arange(50, 500, 25, dtype=int)),
               'subsample': hp.uniform('subsample', 0.5, 1),
               'feature_fraction': hp.uniform('feature_fraction', 0.5, 1), # colsample_bytree
               'reg_alpha': hp.uniform('reg_alpha', 0, 1),
               'reg_lambda':  hp.uniform('reg_lambda', 0, 1),               
               'min_child_samples': hp.choice('min_child_samples',np.arange(10, 100, 10, dtype=int))}

target = "label"
features = [col for col in train.columns if col not in target]

X_train, X_valid, y_train, y_valid = train_test_split(train[features], train[target], test_size=0.2, random_state=849)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid, label=y_valid)
del X_train, y_train
gc.collect()

def evaluate_metric(params):
    print(params)
    model = lgb.train(params, lgb_train,num_boost_round = 3000, valid_sets=[lgb_train, lgb_valid],
                           early_stopping_rounds=100, verbose_eval=200)

    pred = model.predict(X_valid)
    score = -roc_auc_score(y_valid,pred) # Hyperopt tries to minimize error by default
    
    print(score)
    return {'loss': score,'status': STATUS_OK,'stats_running': STATUS_RUNNING}


# Trail
trials = Trials()
# Set algoritm parameters
algo = partial(tpe.suggest, n_startup_jobs=-1)
# Setting the number of evals
MAX_EVALS = 70
best_vals = fmin(evaluate_metric, space=hyper_space, verbose=1, algo=algo, max_evals=MAX_EVALS, trials=trials)
# Print best parameters
best_params = space_eval(hyper_space, best_vals)
print(best_params)