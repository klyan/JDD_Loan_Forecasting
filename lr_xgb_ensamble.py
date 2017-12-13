#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import json
import pandas as pd
import numpy as np
import os
import datetime
import random
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Lasso
import time
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import bisect
from scipy.sparse import coo_matrix


def sparseDfToCsc(df):
    columns = df.columns
    dat, rows = map(list,zip(*[(df[col].sp_values-df[col].fill_value, df[col].sp_index.to_int_index().indices) for col in columns]))
    cols = [np.ones_like(a)*i for (i,a) in enumerate(dat)]
    datF, rowsF, colsF = np.concatenate(dat), np.concatenate(rows), np.concatenate(cols)
    arr = coo_matrix((datF, (rowsF, colsF)), df.shape, dtype=np.float64)
    return arr.tocsc()


def get_threshold(tree_json):
    threshold = []
    if "right_child" in tree_json.keys():
        threshold.extend(get_threshold(tree_json['right_child']))
    if "left_child" in tree_json.keys():
        threshold.extend(get_threshold(tree_json['left_child']))
    if "threshold" in tree_json.keys():
        threshold.extend([tree_json['threshold']])
    return threshold


def lgb_create_features(model, lgbdata, split_raw_data):
    pred_leaf = model.predict(lgbdata, pred_leaf = True)
    pd_pred_leaf = pd.DataFrame(pred_leaf).reset_index(drop=True)
    if category_f is not None and len(category_f) > 0:
        law_cate_feature = pd.get_dummies(lgbdata[category_f],sparse=True,columns=category_f).reset_index(drop=True)  #原始分类型特征dummy
        pd_pred_feature = pd.get_dummies(pd_pred_leaf,sparse=True,columns=pd_pred_leaf.columns).reset_index(drop=True) #GBDT叶子dummy
        newdata = pd.concat([split_raw_data.reset_index(drop=True), pd_pred_feature, law_cate_feature], axis=1, ignore_index=True)
        newdata.columns = split_raw_data.columns.append(pd_pred_feature.columns).append(law_cate_feature.columns)
    else:
        print "do not contains category features"
        pd_pred_feature = pd.get_dummies(pd_pred_leaf,sparse=True,columns=pd_pred_leaf.columns).reset_index(drop=True) #GBDT叶子dummy
        newdata = pd.concat([split_raw_data.reset_index(drop=True), pd_pred_feature], axis=1, ignore_index=True)
        newdata.columns = split_raw_data.columns.append(pd_pred_feature.columns)
    return newdata.fillna(0)

def split_raw_data(reg_data, dvalid_data, dtrain_alldata, dtest_data):
    tmp = pd.DataFrame()
    reg_data["type"] = "train"
    dvalid_data["type"] = "valid"
    dtrain_alldata["type"] = "train_all"
    dtest_data["type"] = "test"
    print "loging"
    all_data = reg_data.append(dvalid_data, ignore_index=True).append(dtrain_alldata, ignore_index=True).append(dtest_data, ignore_index=True)
    gbmreg = lgb.LGBMRegressor(num_leaves = 12, max_depth=3, n_estimators= 1)
    for col in reg_data.columns:
        if col in category_f or col == "type":
            continue
        print col
        gbmreg.fit(pd.DataFrame(dtrain_alldata[col].fillna(0)), tr_user["loan_sum"])
        split = sorted(get_threshold(gbmreg.booster_.dump_model()["tree_info"][0]["tree_structure"]))  
        categories = all_data[col].fillna(0).apply(lambda x: bisect.bisect_left(split, x))
        tmp = pd.concat([tmp,categories],axis=1)
    tmp = pd.get_dummies(tmp,sparse=True,columns=tmp.columns).reset_index(drop=True)
    reg_data.drop("type",axis=1,inplace=True)
    dvalid_data.drop("type",axis=1,inplace=True)
    dtrain_alldata.drop("type",axis=1,inplace=True)
    dtest_data.drop("type",axis=1,inplace=True) 
    return tmp, tmp[all_data["type"] == "train"], tmp[all_data["type"] == "valid"], tmp[all_data["type"]=="train_all"], tmp[all_data["type"] == "test"]


def getPast3MonthLoanPlanInterval(df, duser1, duser2):
    valid_mask = df.month.isin([8,  9, 10])
    test_mask = df.month.isin([9, 10, 11])
    window_size = 90
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask].groupby(["uid","plannum"]).apply(lambda x: x.sort_values(["loan_time"], ascending=True)).reset_index(drop=True)
        tmp["last_sameplan_loantime"] = tmp.groupby(["uid","plannum"])["loan_time"].shift(1)
        tmp["last_sameplan_loan_interval"] = (tmp["loan_time"] - tmp['last_sameplan_loantime']).apply(lambda x:x.days+x.seconds/86400.0)
        ##每期的贷款期数pivot
        perPlanInterval= tmp.groupby(["uid","plannum"])["last_sameplan_loan_interval"].agg(['max','min','mean','median']).reset_index()
        perPlanInterval["plannum"]  =  perPlanInterval['plannum'].astype(str) + "_plannum_" + str(window_size)
        perPlanInterval = perPlanInterval.pivot(index='uid', columns='plannum').reset_index()
        new_list = ["uid"]
        for words in perPlanInterval.columns.get_values():
            if "uid" in words :
                continue
            new_list.append('_'.join(words))
        perPlanInterval.columns =  new_list
        if idx == 0:
            duser1 = duser1.merge(perPlanInterval, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(perPlanInterval, how="left", on="uid")
    return duser1, duser2


#过去一个月三个月，消费、贷款、点击数不为0的天数
tr_user, ts_user = getActionDays(t_order, "buy_time", tr_user, ts_user, 30, "buy")
tr_user, ts_user = getActionDays(t_loan, "date", tr_user, ts_user, 30, "loan")
tr_user, ts_user = getActionDays(t_click, "date", tr_user, ts_user, 30, "click")
tr_user, ts_user = getActionDays(t_loan, "date", tr_user, ts_user, 90, "loan")


##小额、大额借贷的频率（期数？）小期数借款间隔
tr_user, ts_user = getPast3MonthLoanPlanInterval(t_loan, tr_user, ts_user)

#最近一个月贷款比上过去三个月贷款金额的均值
##用户前三个月每个月消费，贷款、金额、次数
tr_user, ts_user = getAmtBeforeRatio(t_loan, "loan_amount", tr_user, ts_user)

#最近一个月消费比上过去三个月消费金额的均值
tr_user, ts_user = getAmtBeforeRatio(t_order, "order_amt", tr_user, ts_user)


to_drop = ["uid","active_date","loan_sum","pay_end_date"]
features = list(np.setdiff1d(tr_user.columns.tolist(), to_drop))
features = tr_user[features].columns.intersection(ts_user.columns) 
category_f = ["sex_age_limit"]#,u'pid_param_pos1', u'pid_param_pos2',u'pid_param_pos3', u'pid_param_pos4', u'pid_param_pos5']
random.seed(888)
select_rows = random.sample(tr_user.index, int(len(tr_user.index)*0.7))
train_df = tr_user.loc[select_rows]
valid_df = tr_user.drop(select_rows)

dtrain = lgb.Dataset(train_df[features], label=train_df["loan_sum"], free_raw_data=False)
dvalid = lgb.Dataset(valid_df[features], label=valid_df["loan_sum"], free_raw_data=False)
dtrain_all = lgb.Dataset(tr_user[features], label=tr_user["loan_sum"], free_raw_data=False)
dtest = lgb.Dataset(ts_user[features], free_raw_data=False)



param = {'num_leaves':8,'num_boost_round':150, 'objective':'regression_l2','metric':'rmse',"learning_rate" : 0.05, "boosting":"gbdt"}
bst = lgb.train(param, dtrain, valid_sets=[dtrain, dvalid],  verbose_eval=100)
print('train mae: %g' % sqrt(mean_squared_error(train_df["loan_sum"], bst.predict(dtrain.data))))
print('valid mae: %g' % sqrt(mean_squared_error(valid_df["loan_sum"], bst.predict(dvalid.data))))


imp = bst.feature_importance(importance_type='gain', iteration=-1)
feat_importance = pd.Series(imp,bst.feature_name()).to_dict()
feat_importance = sorted(feat_importance.iteritems() ,key = lambda asd:asd[1],reverse=True)
imp = pd.DataFrame(feat_importance)

features = list(imp[imp[1] != 0][0])

all_tmp, split_dtrain, split_dvalid, split_all, split_dtest = split_raw_data(dtrain.data[features], dvalid.data[features], dtrain_all.data[features], dtest.data[features])
dtrain.data.drop("type",axis=1,inplace=True)
dvalid.data.drop("type",axis=1,inplace=True)
dtrain_all.data.drop("type",axis=1,inplace=True)
dtest.data.drop("type",axis=1,inplace=True)

lr_train_data = lgb_create_features(bst, dtrain.data, split_dtrain)  #(63695, 66594)
lr_valid_data = lgb_create_features(bst, dvalid.data, split_dvalid)  #(27298, 30197)
lr_all_data = lgb_create_features(bst, dtrain_all.data, split_all)  #(90993, 93892)
lr_test_data = lgb_create_features(bst, dtest.data, split_dtest)




##部分数据训练，
lr = Lasso(alpha=0.003, normalize=False, copy_X=True, max_iter=100000, warm_start =True, precompute=True)
union_feature = list(lr_train_data.columns.intersection(lr_valid_data.columns).intersection(lr_test_data.columns))
model = lr.fit(lr_train_data[union_feature], train_df["loan_sum"])
print "num_iter: ",lr.n_iter_
print "num_coef: ", sum(lr.coef_!=0)
pred_lasso_train = lr.predict(lr_train_data[union_feature])
pred_lasso_valid = lr.predict(lr_valid_data[union_feature])
rmse_lasso_train = sqrt(mean_squared_error(train_df["loan_sum"], pred_lasso_train))
rmse_lasso_valid = sqrt(mean_squared_error(valid_df["loan_sum"], pred_lasso_valid))
print('train mae: %g' %  rmse_lasso_train)
print('valid mae: %g' % rmse_lasso_valid)



##全部数据训练
all_features = list(lr_all_data.columns.intersection(lr_test_data.columns))
lr = Lasso(alpha=0.003, normalize=False, copy_X=True, max_iter=100000, warm_start =True, precompute=True)
model = lr.fit(lr_all_data[all_features], tr_user["loan_sum"])
print "num_iter: ",lr.n_iter_
print "num_coef: ", sum(lr.coef_!=0)
rmse_lasso_train = sqrt(mean_squared_error(tr_user["loan_sum"], lr.predict(lr_all_data[all_features])))
print('train mae: %g' %  rmse_lasso_train)

##保存上传文件
lr_pred = lr.predict(lr_test_data[all_features])
id_test = ts_user['uid']
lr_sub = pd.DataFrame({'uid': id_test, 'loan_sum_lr': lr_pred})
print(lr_sub.describe())
lr_sub.loc[lr_sub["loan_sum_lr"] < 0,"loan_sum_lr"] = 0
now_time = time.strftime("%m-%d %H_%M_%S", time.localtime()) 
lr_sub.to_csv("./submission/" +now_time+'_gbdt_lr_Vscore_'+ str(rmse_lasso_valid) +'.csv', index=False, header=False)
#lr_sub = pd.read_csv("/Users/zhangkai/Desktop/gbdt_lr.csv")


##模型融合
final = lr_sub.merge(lgb_sub,on="uid",how="left")

##手动调参
final["loan_sum"]= 0.3 * final["loan_sum_lr"] + 0.7* final["lgb_loan_sum"]
final.loc[final["loan_sum"] < 0,"loan_sum"] = 0
print(final.describe())
final[["uid","loan_sum"]].to_csv("./submission/" + now_time + 'lightgbm_gbdt_lr.csv', index=False, header=False)


#######xgboost
import xgboost as xgb
import random
random.seed(888)


xgb_dtrain = xgb.DMatrix(train_df[features], label=train_df["loan_sum"])
xgb_dvalid = xgb.DMatrix(valid_df[features], label=valid_df["loan_sum"])
xgb_dtrain_all = xgb.DMatrix(tr_user[features], label=tr_user["loan_sum"])
xgb_dtest = xgb.DMatrix(ts_user[features])

watchlist = [(xgb_dtrain, 'train'), (xgb_dvalid, 'eval')]
param = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'eta': 0.08,
        'num_round': 500, #300
        'max_depth': 3,
        'nthread': -1,
        'seed': 888,
        'silent': 1,
        'lambda':1500,
        'min_child_weight': 4
    }
#{'n_estimators': 100, 'max_depth': 5, }


xgbmodel = xgb.train(param, xgb_dtrain, param['num_round'], watchlist, verbose_eval=100)
rmse_xgb_train = sqrt(mean_squared_error(train_df["loan_sum"], xgbmodel.predict(xgb_dtrain)))
print('train mae: %g' % rmse_xgb_train)
rmse_xgb_valid = sqrt(mean_squared_error(valid_df["loan_sum"], xgbmodel.predict(xgb_dvalid)))
print('valid mae: %g' % rmse_xgb_valid)

xgbmodel = xgb.train(param, xgb_dtrain_all, param['num_round'], verbose_eval=1)
print('valid mae: %g' % sqrt(mean_squared_error(tr_user["loan_sum"], xgbmodel.predict(xgb_dtrain_all))))

 
xgb_pred = xgbmodel.predict(xgb_dtest)
id_test = ts_user['uid']
xgb_sub = pd.DataFrame({'uid': id_test, 'loan_sum_xgb': xgb_pred})
print(xgb_sub.describe())
xgb_sub.loc[xgb_sub["loan_sum_xgb"] < 0,"loan_sum_xgb"] = 0
now_time = time.strftime("%m-%d %H_%M_%S", time.localtime()) 
xgb_sub.to_csv("./submission/" +now_time+'_xgb_Vscore_'+ str(rmse_xgb_valid) +'.csv', index=False, header=False)


##模型融合
final = lr_sub.merge(lgb_sub,on="uid",how="left").merge(xgb_sub, on="uid",how="left")

final = lgb_sub.merge(xgb_sub, on="uid",how="left")

final["loan_sum"]=  0.9* final["lgb_loan_sum"] + 0.1 * final["loan_sum_xgb"]

##手动调参
final["loan_sum"]= 0.3 * final["loan_sum_lr"] + 0.5* final["lgb_loan_sum"] + 0.2 * final["loan_sum_xgb"]
final.loc[final["loan_sum"] < 0,"loan_sum"] = 0
print(final.describe())
final[["uid","loan_sum"]].to_csv("./submission/" + now_time + 'lightgbm_gbdt_lr_xgb.csv', index=False, header=False)



###自动学习融合参数
ensemble_train = pd.DataFrame({'lasso': pred_lasso_train, 'lgb': pred_lgb_train, 'xgb': rmse_xgb_train})
ensemble_valid = pd.DataFrame({'lasso': pred_lasso_valid, 'lgb': pred_lgb_valid, 'xgb': rmse_xgb_valid})
ensemble_test = pd.DataFrame({'lasso': lr_pred, 'lgb': pred, 'xgb': xgb_pred})
ensemble_lr = Lasso(alpha=0.000001, normalize=False, copy_X=True, max_iter=100000, warm_start =True, precompute=True, fit_intercept=False,positive=True)
ensemble_lr.fit(ensemble_train, train_df["loan_sum"])
print "num_iter: ",ensemble_lr.n_iter_
print "coef: ", ensemble_lr.coef_
rmse_ensemble_train = sqrt(mean_squared_error(train_df["loan_sum"], ensemble_lr.predict(ensemble_train)))
rmse_ensemble_valid = sqrt(mean_squared_error(valid_df["loan_sum"], ensemble_lr.predict(ensemble_valid)))
print('train mae: %g' %  rmse_ensemble_train)
print('valid mae: %g' % rmse_ensemble_valid)
final["loan_sum"] = ensemble_lr.coef_[0] * final["loan_sum_lr"] + ensemble_lr.coef_[1] * final["lgb_loan_sum"]


ensemble_model =  lgb.LGBMRegressor(num_leaves = 3, learning_rate = 0.05, n_estimators= 100)
ensemble_model.fit(ensemble_train, train_df["loan_sum"])
rmse_ensemble_train = sqrt(mean_squared_error(train_df["loan_sum"], ensemble_model.predict(ensemble_train)))
rmse_ensemble_valid = sqrt(mean_squared_error(valid_df["loan_sum"], ensemble_model.predict(ensemble_valid)))
print('train mae: %g' %  rmse_ensemble_train)
print('valid mae: %g' % rmse_ensemble_valid)

ensemble_pred = ensemble_model.predict(ensemble_test)
id_test = ts_user['uid']
stacking_sub = pd.DataFrame({'uid': id_test, 'ensemble_pred': ensemble_pred})
print(stacking_sub.describe())
stacking_sub.loc[stacking_sub["ensemble_pred"] < 0,"ensemble_pred"] = 0
now_time = time.strftime("%m-%d %H_%M_%S", time.localtime()) 
stacking_sub[["uid","ensemble_pred"]].to_csv("./submission/" +now_time+'_ensemble_pred.csv', index=False, header=False)




##FM
from fastFM import als
cscMatrix = sparseDfToCsc(lr_train_data[union_feature])
validCscdMatrix = sparseDfToCsc(lr_valid_data[union_feature])
fm = als.FMRegression(n_iter=18, init_stdev=0.1, rank=8, l2_reg_w=40000, l2_reg_V=100000)
fm = als.FMRegression(n_iter=18, init_stdev=0.1, rank=2, l2_reg_w=0.001, l2_reg_V=0.006)

fm.fit(cscMatrix,train_df["loan_sum"])
print(sum(fm.w_ !=0))
print( sqrt(mean_squared_error(train_df["loan_sum"], fm.predict(cscMatrix))))
rmse_fm_valid= sqrt(mean_squared_error(valid_df["loan_sum"], fm.predict(validCscdMatrix)))
print(rmse_fm_valid )


##FM全部数据训练
fm = als.FMRegression(n_iter=18, init_stdev=0.1, rank=2, l2_reg_w=40000, l2_reg_V=100000)
all_features = list(lr_all_data.columns.intersection(lr_test_data.columns))
allcscMatrix = sparseDfToCsc(lr_all_data[all_features])
fm.fit(allcscMatrix, tr_user["loan_sum"])
rmse_fm_train = sqrt(mean_squared_error(tr_user["loan_sum"], fm.predict(allcscMatrix)))
print('train rmse: %g' %  rmse_fm_train)

##FM保存上传文件
testcscMatrix = sparseDfToCsc(lr_test_data[all_features])
fm_pred = fm.predict(testcscMatrix)
id_test = ts_user['uid']
fm_sub = pd.DataFrame({'uid': id_test, 'loan_sum_fm': fm_pred})
print(fm_sub.describe())
fm_sub.loc[fm_sub["loan_sum_fm"] < 0,"loan_sum_fm"] = 0
now_time = time.strftime("%m-%d %H_%M_%S", time.localtime()) 
fm_sub[["uid","loan_sum_fm"]].to_csv("./submission/" +now_time+'_gbdt_fm_Vscore_'+ str(rmse_fm_valid)+'.csv', index=False, header=False)
#lr_sub = pd.read_csv("/Users/zhangkai/Desktop/gbdt_lr.csv")



#####Random Forest
from sklearn.ensemble import RandomForestRegressor
#[m/3]
regr = RandomForestRegressor(n_estimators= 100, max_depth=3, criterion= "mae", random_state=0, n_jobs = 30, oob_score =False, max_features= "log2", warm_start = True)
rf_feature = list(set(features) - set(category_f))

indices_to_keep = ~x_train[rf_feature].isin([np.nan, np.inf, -np.inf]).any(1)

regr.fit(x_train[rf_feature].replace([np.inf,np.nan], 0), x_train["delivery_duration"])

rf_train = regr.predict(x_train[rf_feature].fillna(0))
rf_test = regr.predict(x_test[rf_feature].fillna(0))
rf_valid = regr.predict(x_valid[rf_feature].fillna(0))


print('train mae: %g' % np.mean(np.abs((np.power(2,x_train["delivery_duration"]) -1) - (np.power(2,rf_train) -1) )))
print('valid mae: %g' % np.mean(np.abs((np.power(2,x_valid["delivery_duration"])-1) - (np.power(2, rf_valid) -1) )))
test_mae1 = np.mean(np.abs((np.power(2,x_test["delivery_duration"])-1) - (np.power(2, rf_test) -1) ))
print('test mae: %g' % test_mae1)


regr.feature_importances_,rf_feature

feat_importance = pd.Series(regr.feature_importances_,rf_feature).to_dict()
feat_importance = sorted(feat_importance.iteritems() ,key = lambda asd:asd[1],reverse=True)
imp = pd.DataFrame(feat_importance)






####
##zscore
z_dtrain = copy.deepcopy(train_df[features])
z_dvalid = copy.deepcopy(valid_df[features])
z_dtrain_all = copy.deepcopy(tr_user[features])
z_dtest = copy.deepcopy(ts_user[features])

for col in set(z_dvalid.columns) - set(category_f) -set(["sex"]) - set(to_drop):
    col_zscore = col + '_zscore'
    z_dtrain_all[col_zscore] = ((z_dtrain_all[col] - z_dtrain_all[col].mean())/z_dtrain_all[col].std(ddof=0)).replace([np.inf, -np.inf, np.nan], 0)
    z_dtest[col_zscore] = ((z_dtest[col] - z_dtest[col].mean())/z_dtest[col].std(ddof=0)).replace([np.inf, -np.inf, np.nan], 0)
    z_dtrain[col_zscore] = ((z_dtrain[col] - z_dtrain_all[col].mean())/z_dtrain_all[col].std(ddof=0)).replace([np.inf, -np.inf, np.nan], 0)
    z_dvalid[col_zscore] = ((z_dvalid[col] - z_dtrain_all[col].mean())/z_dtrain_all[col].std(ddof=0)).replace([np.inf, -np.inf, np.nan], 0)

col_zscore = []
for i in z_dtrain_all.columns:
    if '_zscore' in i:
        col_zscore.extend([i])

lr_zscore_train = pd.concat([lr_train_data[union_feature].reset_index(drop=True), z_dtrain[col_zscore].reset_index(drop=True)], axis=1)
lr_zscore_valid = pd.concat([lr_valid_data[union_feature].reset_index(drop=True), z_dvalid[col_zscore].reset_index(drop=True)], axis=1)

lr = Lasso(alpha=0.003, normalize=False, copy_X=True, max_iter=100000, warm_start =True, precompute=True)
model = lr.fit(lr_zscore_train, train_df["loan_sum"])
print "num_iter: ",lr.n_iter_
print "num_coef: ", sum(lr.coef_!=0)
print('train mae: %g' % sqrt(mean_squared_error(train_df["loan_sum"], lr.predict(lr_zscore_train))))
print('valid mae: %g' % sqrt(mean_squared_error(valid_df["loan_sum"], lr.predict(lr_zscore_valid))))
