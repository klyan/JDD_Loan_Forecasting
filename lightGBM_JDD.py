#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import json
import pandas as pd
import numpy as np
import os
import datetime
import math
import random
import xgboost as xgb
import lightgbm as lgb
import time
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


to_drop = ["uid","active_date","loan_sum","pay_end_date",'cate_id_cnt_cate_id_24', 'cate_id_cnt_cate_id_8', 'cate_id_sum_cate_id_24','cate_id_sum_cate_id_8','isloan_pred','clickLoanBehaviorCode_y',u'pid_param_pos1', u'pid_param_pos2', u'pid_param_pos3', u'pid_param_pos4', u'pid_param_pos5']

features = list(np.setdiff1d(tr_user.columns.tolist(), to_drop))
features = tr_user[features].columns.intersection(ts_user.columns) 
random.seed(888)
select_rows = random.sample(tr_user.index, int(len(tr_user.index)*0.7))
train_df = tr_user.loc[select_rows]
valid_df = tr_user.drop(select_rows)


dtrain = lgb.Dataset(train_df[features], label=train_df["loan_sum"],free_raw_data=False)
dvalid = lgb.Dataset(valid_df[features], label=valid_df["loan_sum"], free_raw_data=False)
dtrain_all = lgb.Dataset(tr_user[features], label=tr_user["loan_sum"], free_raw_data=False)
dtest = lgb.Dataset(ts_user[features], free_raw_data=False)

param = {'num_leaves':8,'num_boost_round':500, 'objective':'regression_l2','metric':'rmse',"learning_rate" : 0.05, "boosting":"gbdt", "lambda_l2":1500, "feature_fraction":0.9, "bagging_fraction":0.9, "bagging_freq" : 50, "top_rate": 0.01}

bst = lgb.train(param, dtrain, valid_sets=[dtrain, dvalid],  verbose_eval=100)
pred_lgb_train = bst.predict(dtrain.data)
pred_lgb_valid = bst.predict(dvalid.data)
pred_lgb_all = bst.predict(dtrain_all.data)
print('all rmse: %g' % sqrt(mean_squared_error(tr_user["loan_sum"], pred_lgb_all)))
print('train rmse: %g' % sqrt(mean_squared_error(train_df["loan_sum"], pred_lgb_train)))
valid_score = sqrt(mean_squared_error(valid_df["loan_sum"], pred_lgb_valid))
print('valid rmse: %g' % valid_score)


imp = bst.feature_importance(importance_type='gain', iteration=-1)
feat_importance = pd.Series(imp,bst.feature_name()).to_dict()
feat_importance = sorted(feat_importance.iteritems() ,key = lambda asd:asd[1],reverse=True)
imp = pd.DataFrame(feat_importance)
print(imp)

#logfeatures = list(imp[imp[1] != 0][0])

param = {'num_leaves':8,'num_boost_round':500, 'objective':'regression_l2','metric':'rmse',"learning_rate" : 0.05, "boosting":"gbdt", "lambda_l2":1500, "feature_fraction":0.9, "bagging_fraction":0.9, "bagging_freq" : 50} 
bst = lgb.train(param, dtrain_all, valid_sets=[dtrain_all, dtrain],  verbose_eval=100)
print('feature importance...')


##提交文件
pred = bst.predict(dtest.data)
id_test = ts_user['uid']
lgb_sub = pd.DataFrame({'uid': id_test, 'lgb_loan_sum': pred})
print(lgb_sub.describe())
lgb_sub.loc[lgb_sub["lgb_loan_sum"] < 0,"lgb_loan_sum"] = 0
print('saving submission...')
now_time = time.strftime("%m-%d %H_%M_%S", time.localtime()) 
lgb_sub[["uid","lgb_loan_sum"]].to_csv("./submission/" +now_time+'_lightgbm_Vscore_' + str(valid_score) + '.csv', index=False, header=False)





from catboost import Pool, CatBoostRegressor

train_pool = Pool(train_df[features], train_df["loan_sum"])
test_pool = Pool(valid_df[features], valid_df["loan_sum"]) 
dtrain_all_pool = Pool(tr_user[features], tr_user["loan_sum"])
dtest_pool = Pool(ts_user[features])

catb = CatBoostRegressor(iterations=300, depth=3, learning_rate=0.05, loss_function='RMSE')
catb.fit(train_pool)
print('catb train rmse: %g' % sqrt(mean_squared_error(train_df["loan_sum"], catb.predict( train_pool))))
valid_score = sqrt(mean_squared_error(valid_df["loan_sum"], catb.predict( Pool(valid_df[features]))))
print('catb valid rmse: %g' % valid_score)

catb = CatBoostRegressor(iterations=300, depth=3, learning_rate=0.05, loss_function='RMSE')
catb.fit(dtrain_all_pool)

##提交文件
pred = catb.predict(dtest_pool)
id_test = ts_user['uid']
catb_sub = pd.DataFrame({'uid': id_test, 'catb_loan_sum': pred})
print(catb_sub.describe())
catb_sub.loc[catb_sub["catb_loan_sum"] < 0,"catb_loan_sum"] = 0
print('saving submission...')
now_time = time.strftime("%m-%d %H_%M_%S", time.localtime()) 
catb_sub[["uid","catb_loan_sum"]].to_csv("./submission/" +now_time+'_lightgbm_Vscore_' + str(valid_score) + '.csv', index=False, header=False)



##相关性分析
corr = tr_user[features].corr()
corr["loan_sum"] = corr["loan_sum"].apply(lambda x: abs(x))


#真是结果分布
plt.figure()
tr_user["loan_sum"].hist(bins=20).plot()
plt.show() 





