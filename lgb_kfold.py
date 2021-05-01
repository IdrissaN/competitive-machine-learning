# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:51:29 2020

@author: Idrissa
"""

import gc 
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

def train_loop(df, num_folds, useful_features, target, params, num_boost_round, seed=849):
    kfold = KFold(n_splits = num_folds, shuffle=True, random_state = seed)
    oof_predictions = np.zeros((df.shape[0]))
    clfs = []
    feature_importance = pd.DataFrame()
    fold = 0
    for train_index, valid_index in kfold.split(df):

        print(f"------ Fold {fold+1} started at {time.ctime()} --------")

        x_train = df[useful_features].loc[train_index].copy()
        x_valid = df[useful_features].loc[valid_index].copy()
        y_train = df[target].loc[train_index]
        y_valid = df[target].loc[valid_index]

        tr_data = lgb.Dataset(x_train, label=y_train)
        vl_data = lgb.Dataset(x_valid, label=y_valid)  

        estimator = lgb.train(params, tr_data, valid_sets = [tr_data, vl_data],
                              num_boost_round=num_boost_round, verbose_eval=100, early_stopping_rounds=50) 

        clfs.append(estimator)
        oof_pred = estimator.predict(x_valid)
        oof_predictions[valid_index] = oof_pred

        oof_score = roc_auc_score(y_valid,oof_pred)
        
        imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),x_train.columns)), columns=["importance","feature"])                
        imp["fold"] = fold
        feature_importance = pd.concat([feature_importance, imp], axis=0)
        fold += 1  

    oof_score = roc_auc_score(df[target], oof_predictions)
    print("OOF Score", oof_score)
    return clfs, oof_predictions, feature_importance
    
lgb_params = {'objective':'cross_entropy','boosting_type':'gbdt','metric':'auc','nthread':-1,'learning_rate':0.01,'tree_learner':'serial',
              'num_leaves': 15,'colsample_bytree': 0.7,'min_data_in_leaf': 150,'max_depth':-1,'subsample_freq':1,'subsample':0.8,'max_bin':255,'verbose':-1,'seed':849}   

clfs, oof_predictions, feature_importance = train_loop(df=data, num_folds=5, useful_features=use_cols_final, target = "deriv_is_sale", params=lgb_params,num_boost_round=10000)