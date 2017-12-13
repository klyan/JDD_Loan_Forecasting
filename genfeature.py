#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import json
import pandas as pd
import numpy as np
import os
import datetime
import time
import math
import random
import copy
from datetime import datetime, timedelta,date
import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
import lightgbm as lgb
import time
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


def parseDate(df, col):
	df[col] = pd.to_datetime(df[col])
	if col == "click_time" or col == "loan_time":
		df["date"] = df[col].apply(lambda x: x.date())
	return df


def get_windows_mask(df, time_col, window_size):
	valid_end_date = "2016-11-01"
	test_end_date = "2016-12-01"
	valid_start = pd.Timestamp(valid_end_date) - timedelta(days=window_size)
	valid_mask = (df[time_col] >= valid_start) & (df[time_col] < pd.Timestamp(valid_end_date))
	test_start = pd.Timestamp(test_end_date) - timedelta(days=window_size)
	test_mask = (df[time_col] >= test_start) & (df[time_col] < pd.Timestamp(test_end_date))
	return valid_mask, test_mask

def click_percent(df, duser, tuser, window_size):
	valid_mask, test_mask = get_windows_mask(df, "click_time", window_size)
	for idx, mask in enumerate([valid_mask, test_mask]):
		tmp = df[mask]
		uid_clicks = tmp.groupby(["uid","pid"]).click_time.count().reset_index()
		uid_clicks.columns = ["uid","pid", "clicks"]
		pid_avg_clicks = uid_clicks.groupby(["pid"]).clicks.mean().reset_index()
		pid_avg_clicks.columns = ["pid","avg_clicks"]
		uid_clicks = uid_clicks.merge(pid_avg_clicks,how="left",on="pid")
		uid_clicks["clicks_percent"] = uid_clicks["clicks"]/uid_clicks["avg_clicks"]
		uid_clicks["pid"]  =  uid_clicks['pid'].astype(str) + "_pidcliks_" + str(window_size) + "days"
		if idx == 0:
			uid_clicks = uid_clicks.pivot(index='uid', columns='pid', values='clicks_percent').reset_index().fillna(0)
			duser = duser.merge(uid_clicks, how="left", on="uid").fillna(0)
		elif idx == 1:
			uid_clicks = uid_clicks.pivot(index='uid', columns='pid', values='clicks_percent').reset_index().fillna(0)
			tuser = tuser.merge(uid_clicks, how="left", on="uid").fillna(0)			
	return duser, tuser


def click_days_pids(df, duser, tuser, window_size):
	valid_mask, test_mask = get_windows_mask(df, "click_time", window_size)
	for idx, mask in enumerate([valid_mask, test_mask]):
		tmp = df[mask]
		uid_clicks = tmp.groupby(["uid"])["date","pid"].nunique().reset_index()
		uid_clicks.columns = ["uid","clickdays_" + str(window_size) + "_days", "pids_" + str(window_size)+"days"]
		if idx == 0:
			duser = duser.merge(uid_clicks, how="left", on="uid").fillna(0)
		elif idx == 1:
			tuser = tuser.merge(uid_clicks, how="left", on="uid").fillna(0)
	return duser, tuser


def getNearestClick(df, duser, tuser, window_size):
	offset = 1
	df['last_click_time'] = df.groupby(['uid'])[['click_time']].shift(offset)
	df["click_interval"] = (df["click_time"] - df['last_click_time']).apply(lambda x: x.total_seconds()).fillna(0)
	df['last_click_pid'] = df.groupby(['uid'])['pid'].shift(offset).fillna(0)
	uid_mean_click_interval = df.groupby(["uid"])["click_interval"].mean().reset_index()
	valid_mask, test_mask = get_windows_mask(df, "click_time", window_size)
	for idx, mask in enumerate([valid_mask, test_mask]):
		tmp = df[mask]
		uid_nearest_click = tmp.groupby("uid")["click_time"].max().reset_index()
		uid_click_interval = uid_mean_click_interval.merge(uid_nearest_click,how="left", on="uid")
		if idx == 0:
			uid_click_interval["click_nearest_interval"] = (pd.Timestamp(valid_end_date) - uid_click_interval["click_time"]).apply(lambda x:x.total_seconds())
		elif idx == 1:
			uid_click_interval["click_nearest_interval"] = (pd.Timestamp(test_end_date) - uid_click_interval["click_time"]).apply(lambda x:x.total_seconds())
		uid_click_interval["next_click"] = uid_click_interval["click_interval"] + uid_click_interval["click_nearest_interval"]
		uid_click_interval.drop("click_time",axis=1,inplace=True)
		uid_click_interval.columns = ["uid","mean_click_interval", "click_nearest_interval", "nextclicktime"]
		if idx == 0:
			duser = duser.merge(uid_click_interval,how="left",on="uid")
		elif idx == 1:
			tuser = tuser.merge(uid_click_interval,how="left",on="uid")
	return duser, tuser


def uid_order_status(df, duser, tuser, window_size):
	valid_mask, test_mask = get_windows_mask(df, "buy_time", window_size)
	for idx, mask in enumerate([valid_mask, test_mask]):
		tmp = df[mask]
		cate_total_sale_amt = tmp.groupby(["uid"])["order_amt"].sum().reset_index()
		cate_total_sale_cnt = tmp.groupby(["uid"])["buy_time"].count().reset_index()
		uid_order = cate_total_sale_amt.merge(cate_total_sale_cnt, how="left", on =["uid"])
		uid_order["avg_order_amt"] = uid_order["order_amt"]/ uid_order["buy_time"]
		uid_order["mean_order_amt_percent"] = uid_order["order_amt"] / np.mean(uid_order["order_amt"])
		uid_order["mean_buy_cnt_percent"] = uid_order["buy_time"] / np.mean(uid_order["buy_time"])
		uid_order["mean_order_amt_percent_mean_buy_cnt_percent"] = uid_order["mean_order_amt_percent"] * uid_order["mean_buy_cnt_percent"]
		uid_order.columns = ['uid', 'order_amt'+str(window_size), 'buy_cnt'+str(window_size), "avg_order_amt" + str(window_size) , 'mean_order_amt_percent'+str(window_size), 'mean_buy_cnt_percent'+str(window_size), 'mean_order_amt_percent_mean_buy_cnt_percent'+str(window_size)]
		if idx == 0:
			duser = duser.merge(uid_order, how="left", on = 'uid')
		elif idx == 1:
			tuser = tuser.merge(uid_order, how="left", on = 'uid')
	return duser, tuser




def getNearestOrder(df, duser, tuser, window_size):
	offset = 1
	df['last_buy_time'] = df.groupby(['uid'])[['buy_time']].shift(offset)
	df["buy_interval"] = (df["buy_time"] - df['last_buy_time']).apply(lambda x: x.days).fillna(0)
	uid_buy_interval = df.groupby(["uid"])["buy_interval"].mean().reset_index()
	#df.drop("buy_interval", inplace=True, axis=1)
	valid_mask, test_mask = get_windows_mask(df, "buy_time", window_size)
	for idx, mask in enumerate([valid_mask, test_mask]):
		tmp = df[mask]
		tmp = tmp.groupby(['uid','buy_time'])["order_amt"].sum().reset_index()  #每个人一天内消费了多少金额
		maxtime_idx = tmp.groupby(['uid'])['buy_time'].transform(max) == tmp['buy_time']  #用户最近一天消费的情况
		uid_nearest_buy = tmp[maxtime_idx]
		uid_nearest_buy = uid_buy_interval.merge(uid_nearest_buy, on="uid", how="left")
		if idx == 0:
			uid_nearest_buy["buy_nearest_interval"] = (pd.Timestamp(valid_end_date) - uid_nearest_buy["buy_time"]).apply(lambda x:x.days)
		elif idx == 1:
			uid_nearest_buy["buy_nearest_interval"] = (pd.Timestamp(test_end_date) - uid_nearest_buy["buy_time"]).apply(lambda x:x.days)		
		uid_nearest_buy["next_buytime"] = uid_nearest_buy["buy_interval"] + uid_nearest_buy["buy_nearest_interval"]
		uid_nearest_buy["buy_nearest_price_interval"] = uid_nearest_buy["order_amt"] / (uid_nearest_buy["buy_nearest_interval"]+1)
		#uid_nearest_buy.drop(["buy_interval"],axis=1,inplace=True)
		uid_nearest_buy.columns = ["uid", "mean_buy_interval", "nearest_buytime", "buy_nearest_price", "buy_nearest_interval", "next_buytime", "buy_nearest_price_interval"]
		if idx == 0:
			duser = duser.merge(uid_nearest_buy, how="left", on="uid")
			duser["nearest_buytime_to_active_date"] = (duser["nearest_buytime"] - duser["active_date"]).apply(lambda x: x.days)#最近一次购买距离用户激活的时间
			duser.drop("nearest_buytime", inplace=True, axis=1)
		elif idx == 1:
			tuser = tuser.merge(uid_nearest_buy, how="left", on="uid")
			tuser["nearest_buytime_to_active_date"] = (tuser["nearest_buytime"] - tuser["active_date"]).apply(lambda x: x.days)#最近一次购买距离用户激活的时间
			tuser.drop("nearest_buytime",inplace=True,axis=1)
	return duser, tuser

def getMaxPriceOrder(df, duser, tuser, window_size):
	valid_mask, test_mask = get_windows_mask(df, "buy_time", window_size)
	for idx, mask in enumerate([valid_mask, test_mask]):
		tmp = df[mask]
		tmp = tmp.groupby(['uid','buy_time'])["order_amt"].sum().reset_index()  #每个人一天内消费了多少金额
		tmp["max_order_amt"] = tmp.groupby(['uid'])['order_amt'].transform(max)  #用户消费最大的情况
		uid_max_buy = tmp[(tmp["max_order_amt"] == tmp["order_amt"])]
		max_amt_idx = uid_max_buy.groupby("uid")['buy_time'].transform(max) == uid_max_buy['buy_time']
		uid_max_buy = uid_max_buy[max_amt_idx]
		uid_max_buy.drop("max_order_amt",inplace=True,axis=1)
		if idx == 0:
			uid_max_buy["maxbuy_interval"] = (pd.Timestamp(valid_end_date) - uid_max_buy["buy_time"]).apply(lambda x:x.days)
		elif idx == 1:
			uid_max_buy["maxbuy_interval"] = (pd.Timestamp(test_end_date) - uid_max_buy["buy_time"]).apply(lambda x:x.days)	
		uid_max_buy["maxbuy_price_interval"] = uid_max_buy["order_amt"] / (uid_max_buy["maxbuy_interval"]+1)
		uid_max_buy.drop(["buy_time"],axis=1,inplace=True)
		uid_max_buy.columns = ["uid","maxbuy_price", "maxbuy_interval", "maxbuy_price_interval"]
		if idx == 0:
			duser = duser.merge(uid_max_buy,how="left",on="uid")
		elif idx == 1:
			tuser = tuser.merge(uid_max_buy,how="left",on="uid")
	return duser, tuser



def gen_fixedtw_features_for_loan(df, duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(df, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask]
        #贷款金额 loan_amount
        stat_loanAmt = tmp.groupby(["uid"])['loan_amount'].agg(['sum','count','mean','max','min']).reset_index()
        stat_loanAmt.columns=['uid']+ [i+ '_loanAmt_'+str(window_size) for i in list(stat_loanAmt.columns)[1:]]
        #贷款期数 plannum   #平均值去掉?
        stat_loanPlanNum=tmp.groupby(["uid"])['plannum'].agg(['mean','max','min']).reset_index()
        stat_loanPlanNum.columns=['uid']+ [i+ '_loanPlanNum_'+str(window_size) for i in list(stat_loanPlanNum.columns)[1:]]
        #每期贷款额 amt_per_plan
        stat_amtPerPlan=tmp.groupby(["uid"])['amt_per_plan'].agg(['sum','mean','max','min']).reset_index()
        stat_amtPerPlan.columns = ['uid']+ [i+ '_amtPerPlan_'+str(window_size) for i in list(stat_amtPerPlan.columns)[1:]]
        #频率最高的贷款期数和对应的贷款次数
        freq_plannum=tmp.groupby('uid').plannum.value_counts().rename('freq_plannum').reset_index()
        idx_mostxfreq=list(freq_plannum.groupby('uid').freq_plannum.idxmax())
        most_freq=freq_plannum.loc[idx_mostxfreq]
        most_freq.columns=['uid','most_plannum_'+str(window_size),'freq_most_plannum_'+str(window_size)]
        ##每期的贷款期数pivot
        perPlanAmtCnt= tmp.groupby(["uid","plannum"])["loan_amount"].agg(['count','sum']).reset_index()
        perPlanAmtCnt["plannum"]  =  perPlanAmtCnt['plannum'].astype(str) + "_plannum_" + str(window_size) + "days"
        perPlanAmtCnt = perPlanAmtCnt.pivot(index='uid', columns='plannum').reset_index().fillna(0)
        new_list = ["uid"]
        for words in perPlanAmtCnt.columns.get_values():
            if "uid" in words :
                continue
            new_list.append('_'.join(words))
        perPlanAmtCnt.columns =  new_list
        #贷款周期 loan_interval
        stat_loanInterval=tmp.groupby(["uid"])['loan_interval'].agg(['mean','median','max','min']).reset_index()
        stat_loanInterval.columns=['uid']+ [i+ '_loanInterval_'+str(window_size) for i in list(stat_loanInterval.columns)[1:]]
        loan_stat = stat_loanAmt.merge(stat_loanPlanNum,  how="left", on="uid").merge(stat_amtPerPlan, how="left", on="uid").merge(stat_loanInterval, how="left", on="uid").merge(most_freq, how="left", on="uid").merge(perPlanAmtCnt, how="left", on="uid")
        if idx==0:
            duser1=duser1.merge(loan_stat, how="left", on="uid")
            duser1[new_list] = duser1[new_list].fillna(0.0)
            stat_daysLoan=(pd.Timestamp(valid_end_date)-tmp.loan_time).apply(lambda x:x.days+x.seconds/86400.0).groupby(tmp.uid).agg(['mean','max','min']).reset_index()     #各次贷款离现在的时间相关的统计
            stat_daysLoan.columns = ['uid']+ [i+ '_nearestLoanInterval_'+str(window_size) for i in list(stat_daysLoan.columns)[1:]]
            duser1=duser1.merge(stat_daysLoan, how="left", on="uid")
        elif idx==1:
             duser2=duser2.merge(loan_stat, how="left", on="uid")
             duser2[new_list] = duser2[new_list].fillna(0.0)
             stat_daysLoan=(pd.Timestamp(test_end_date)-tmp.loan_time).apply(lambda x:x.days+x.seconds/86400.0).groupby(tmp.uid).agg(['mean','max','min']).reset_index() 
             stat_daysLoan.columns = ['uid']+ [i+ '_nearestLoanInterval_'+str(window_size) for i in list(stat_daysLoan.columns)[1:]]
             duser2=duser2.merge(stat_daysLoan, how="left", on="uid")
    return duser1, duser2


def getNearestLoan(df, duser1, duser2):
    valid_mask = df.month.isin([8,  9, 10])
    test_mask = df.month.isin([8, 9, 10, 11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask]
        maxtime_idx = tmp.groupby(['uid'])['loan_time'].transform(max) == tmp['loan_time']  #用户最近一次贷款的情况
        uid_nearest_loan = tmp[maxtime_idx].reset_index(drop=True)
        if idx==0:
            uid_nearest_loan['nearest_loantime']=(pd.Timestamp(valid_end_date)-uid_nearest_loan.loan_time).apply(lambda x:x.days+x.seconds/86400.0)
            uid_nearest_loan['nearest_loan_amt_time'] = uid_nearest_loan['loan_amount'] / (1 + uid_nearest_loan['nearest_loantime']) 
            uid_nearest_loan = uid_nearest_loan[["uid", "plannum", "amt_per_plan", "loan_amount","nearest_loan_amt_time", "nearest_loantime"]]
            uid_nearest_loan.columns = ["uid","nearest_plannum", "nearest_amt_per_plan", "nearest_loan_amount","nearest_loan_amt_time","nearest_loantime"]
            duser1 = duser1.merge(uid_nearest_loan, how="left", on="uid")
        elif idx==1:
            uid_nearest_loan['nearest_loantime']=(pd.Timestamp(test_end_date)-uid_nearest_loan.loan_time).apply(lambda x:x.days+x.seconds/86400.0)
            uid_nearest_loan['nearest_loan_amt_time'] = uid_nearest_loan['loan_amount'] / (1 + uid_nearest_loan['nearest_loantime']) 
            uid_nearest_loan = uid_nearest_loan[["uid", "plannum", "amt_per_plan", "loan_amount","nearest_loan_amt_time","nearest_loantime"]]
            uid_nearest_loan.columns = ["uid","nearest_plannum", "nearest_amt_per_plan", "nearest_loan_amount","nearest_loan_amt_time","nearest_loantime"]
            duser2 = duser2.merge(uid_nearest_loan, how="left", on="uid")
    return duser1, duser2


def current2PayAmt(df, duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(df, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        if idx == 0:
            pay_date_mask = df["pay_end_date"] > pd.Timestamp(valid_end_date)
            tmp = df[mask & pay_date_mask]
            current_pay_amt = tmp.groupby("uid")["amt_per_plan"].agg(["sum"]).reset_index()
            current_pay_amt.columns = ["uid", "current_topay_amt"]
            duser1 = duser1.merge(current_pay_amt, on="uid", how="left")
            duser1["current_topay_amt"] = duser1["current_topay_amt"].fillna(0.0)
        elif idx == 1:
            pay_date_mask = df["pay_end_date"] > pd.Timestamp(test_end_date)
            tmp = df[mask & pay_date_mask]
            current_pay_amt = tmp.groupby("uid")["amt_per_plan"].agg(["sum"]).reset_index()
            current_pay_amt.columns = ["uid", "current_topay_amt"]
            duser2 = duser2.merge(current_pay_amt, on="uid", how="left")
            duser2["current_topay_amt"] = duser2["current_topay_amt"].fillna(0.0)
    return duser1, duser2


def currentDebtAmt(df, duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(df, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        if idx == 0:
            pay_date_mask = df["pay_end_date"] > pd.Timestamp(valid_end_date)
            tmp = df[mask & pay_date_mask].reset_index(drop=True)
            tmp["payed_num"] = (pd.Timestamp(valid_end_date) - tmp["loan_time"]).apply(lambda x: x.days/30.0)
            tmp["debtAmt"] = tmp["loan_amount"] - tmp["amt_per_plan"] * tmp["payed_num"]
            current_debtAmt = tmp.groupby("uid")["debtAmt"].sum().rename("current_debtAmt_" + str(window_size)).reset_index()
            duser1 = duser1.merge(current_debtAmt, on="uid", how="left")
            duser1["current_debtAmt_" + str(window_size)] = duser1["current_debtAmt_" + str(window_size)].fillna(0.0)
            duser1["remainingAmt_" + str(window_size)] = duser1["limit"]- duser1["current_debtAmt_" + str(window_size)]
        elif idx == 1:
            pay_date_mask = df["pay_end_date"] > pd.Timestamp(valid_end_date)
            tmp = df[mask & pay_date_mask].reset_index(drop=True)
            tmp["payed_num"] = (pd.Timestamp(test_end_date) - tmp["loan_time"]).apply(lambda x: x.days/30.0)
            tmp["debtAmt"] = tmp["loan_amount"] - tmp["amt_per_plan"] * tmp["payed_num"]
            current_debtAmt = tmp.groupby("uid")["debtAmt"].sum().rename("current_debtAmt_" + str(window_size)).reset_index()
            duser2 = duser2.merge(current_debtAmt, on="uid", how="left")
            duser2["current_debtAmt_" + str(window_size)] = duser2["current_debtAmt_" + str(window_size)].fillna(0.0)
            duser2["remainingAmt_" + str(window_size)] = duser2["limit"]- duser2["current_debtAmt_" + str(window_size)]    
    return duser1, duser2

##每人购买力，预测贷款金额
def avgLoanAmt4orderAmt(df_loan, df_order,  duser1, duser2):
    df_order_tmp = df_order[df_order["month"] < 11] 
    df_loan_tmp = df_loan[df_loan["month"] < 11] 
    month_orderAmt = df_order_tmp.groupby(["uid","month"])["order_amt"].sum().rename("uid_month_orderAmt").reset_index().groupby("uid")["uid_month_orderAmt"].mean().rename("uid_avg_month_orderAmt").reset_index()
    month_loanAmt = df_loan_tmp.groupby(["uid","month"])["loan_amount"].sum().rename("uid_month_loanAmt").reset_index().groupby("uid")["uid_month_loanAmt"].mean().rename("uid_avg_month_loanAmt").reset_index()
    month_order_loan = month_orderAmt.merge(month_loanAmt, on = "uid", how="left").fillna(0.0)
    month_order_loan["loanAmt_ratio"] = month_order_loan["uid_avg_month_loanAmt"]/ month_order_loan["uid_avg_month_orderAmt"]  
    duser1 = duser1.merge(month_order_loan, on = 'uid', how="left")
    duser2 = duser2.merge(month_order_loan, on = 'uid', how="left")
    duser1["pred_loanAmt"] = duser1["loanAmt_ratio"] * duser1['order_amt30']
    duser2["pred_loanAmt"] = duser2["loanAmt_ratio"] * duser2['order_amt30']
    duser1.drop(["uid_avg_month_loanAmt","uid_avg_month_orderAmt"], axis=1, inplace=True)
    duser2.drop(["uid_avg_month_loanAmt","uid_avg_month_orderAmt"], axis=1, inplace=True)
    return duser1, duser2


def getLoanAmtRemainingLimt(df_loan, duser1, duser2, before_month):
    valid_start = pd.Timestamp(valid_end_date) - timedelta(days=before_month*31)
    valid_end = pd.Timestamp(valid_end_date) - timedelta(days=(before_month-1)*31)
    test_start = pd.Timestamp(test_end_date) - timedelta(days=before_month*31)
    test_end = pd.Timestamp(test_end_date) - timedelta(days=(before_month-1)*31)
    valid_mask = (df_loan["loan_time"] >= valid_start) & (df_loan["loan_time"] < valid_end)
    test_mask = (df_loan["loan_time"] >= test_start) & (df_loan["loan_time"] < test_end)
    for idx, mask in enumerate([valid_mask, test_mask]):
        uid_month_loanamt = df_loan[mask].groupby("uid")["loan_amount"].sum().rename("month_sum_loanamt"+str(before_month)).reset_index()  #分子
        if idx == 0:
            debt_mask = (df_loan["loan_time"] < valid_start) & (df_loan["pay_end_date"] > valid_start)
            tmp = df_loan[debt_mask].reset_index(drop=True)
            tmp["topay_num"] = (tmp["pay_end_date"] - valid_start).apply(lambda x: x.days/30.0)
            tmp["debtAmt"] = tmp["amt_per_plan"] * tmp["topay_num"]
            current_debtAmt = tmp.groupby("uid")["debtAmt"].sum().rename("debtAmt_monthBefore"+str(before_month)).reset_index()
            current_debtAmt = current_debtAmt.merge(uid_month_loanamt,on="uid",how="left").fillna(0.0)
            duser1 = duser1.merge(current_debtAmt, on="uid", how="left")   #月初的负债额，当月借贷额度
            duser1["debtAmt_monthBefore"+str(before_month)] = duser1["debtAmt_monthBefore"+str(before_month)].fillna(0.0)
            duser1["remainingAmt_monthBefore" + str(before_month)] = duser1["limit"]- duser1["debtAmt_monthBefore" + str(before_month)]
            duser1["loansum_remainingAmt_ratio_monthBefore" + str(before_month)] = duser1["month_sum_loanamt"+str(before_month)]/(1+duser1["remainingAmt_monthBefore" + str(before_month)])
        elif idx == 1:
            tmp = df_loan[(df_loan["loan_time"] < test_start) & (df_loan["pay_end_date"] > test_start)].reset_index(drop=True)
            tmp["topay_num"] = (tmp["pay_end_date"] - test_start).apply(lambda x: x.days/30.0)
            tmp["debtAmt"] = tmp["amt_per_plan"] * tmp["topay_num"]
            current_debtAmt = tmp.groupby("uid")["debtAmt"].sum().rename("debtAmt_monthBefore"+str(before_month)).reset_index()
            current_debtAmt = current_debtAmt.merge(uid_month_loanamt, on="uid",how="left").fillna(0.0)
            duser2 = duser2.merge(current_debtAmt, on="uid", how="left")   #月初的负债额，当月借贷额度
            duser2["debtAmt_monthBefore"+str(before_month)] = duser2["debtAmt_monthBefore"+str(before_month)].fillna(0.0)
            duser2["remainingAmt_monthBefore" + str(before_month)] = duser2["limit"]- duser2["debtAmt_monthBefore" + str(before_month)]
            duser2["loansum_remainingAmt_ratio_monthBefore" + str(before_month)] = duser2["month_sum_loanamt"+str(before_month)]/(1+duser2["remainingAmt_monthBefore" + str(before_month)])
    return duser1, duser2


def getAvailableLoanAmtLimt(df_loan, duser1, duser2, before_month):
    if before_month == 1:
        valid_start = pd.Timestamp("2016-10-01")
        valid_end = pd.Timestamp("2016-11-01")
        test_start = pd.Timestamp("2016-11-01")
        test_end = pd.Timestamp("2016-12-01")
    elif before_month == 2:
        valid_start = pd.Timestamp("2016-09-01")
        valid_end = pd.Timestamp("2016-10-01")
        test_start = pd.Timestamp("2016-10-01")
        test_end = pd.Timestamp("2016-11-01")
    valid_mask = (df_loan["loan_time"] >= valid_start) & (df_loan["loan_time"] < valid_end)
    test_mask = (df_loan["loan_time"] >= test_start) & (df_loan["loan_time"] < test_end)
    for idx, mask in enumerate([valid_mask, test_mask]):
        if idx == 0:
            debt_mask = (df_loan["loan_time"] < valid_end) & (df_loan["pay_end_date"] > valid_end)  #仍然再还的贷款
            tmp = df_loan[debt_mask].reset_index(drop=True)
            tmp["debtRatio"] = (tmp["pay_end_date"] - valid_end).apply(lambda x: 1  if x.days/30.0 >= 1 else x.days/30.0)
            tmp["unAvailableAmt"] = tmp["loan_amount"] * tmp["debtRatio"]
            unAvailableAmt = tmp.groupby("uid")["unAvailableAmt"].sum().rename("unAvailableAmt_monthBefore"+str(before_month)).reset_index()
            duser1 = duser1.merge(unAvailableAmt, on="uid", how="left")   #当月不可用额度
            duser1["unAvailableAmt_monthBefore"+str(before_month)] = duser1["unAvailableAmt_monthBefore"+str(before_month)].fillna(0.0)
            duser1["availableAmt_monthBefore" + str(before_month)] = duser1["limit"]- duser1["unAvailableAmt_monthBefore"+str(before_month)]
        elif idx == 1:
            tmp = df_loan[(df_loan["loan_time"] < test_end) & (df_loan["pay_end_date"] > test_end)].reset_index(drop=True)
            tmp["debtRatio"] = (tmp["pay_end_date"] - test_end).apply(lambda x: 1  if x.days/30.0 >= 1 else x.days/30.0)
            tmp["unAvailableAmt"] = tmp["loan_amount"] * tmp["debtRatio"]
            unAvailableAmt = tmp.groupby("uid")["unAvailableAmt"].sum().rename("unAvailableAmt_monthBefore"+str(before_month)).reset_index()
            duser2 = duser2.merge(unAvailableAmt, on="uid", how="left")   #当月不可用额度
            duser2["unAvailableAmt_monthBefore"+str(before_month)] = duser2["unAvailableAmt_monthBefore"+str(before_month)].fillna(0.0)
            duser2["availableAmt_monthBefore" + str(before_month)] = duser2["limit"]- duser2["unAvailableAmt_monthBefore"+str(before_month)]
    return duser1, duser2

def lastMonthPayedAmt(df, duser1, duser2):
    valid_start = pd.Timestamp(valid_end_date) - timedelta(days=30)
    valid_mask = (df["pay_end_date"] >= valid_start) & (df["loan_time"] < valid_start)
    test_start = pd.Timestamp(test_end_date) - timedelta(days=30)
    test_mask = (df["pay_end_date"] >= test_start) & (df["loan_time"] < test_start)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask].reset_index(drop=True)
        if idx==0:
            tmp["last_month_topay_num"] = tmp["pay_end_date"].apply(lambda x: 1 if (x- valid_start).days/30.0 >=1 else (x- valid_start).days/30.0)
            tmp["last_month_payed_amt"] = tmp["amt_per_plan"] * tmp["last_month_topay_num"]
            lastMonthPayedAmt= tmp.groupby("uid")["last_month_payed_amt"].sum().rename("lastMonthPayedAmt").reset_index()
            duser1 = duser1.merge(lastMonthPayedAmt, on="uid", how="left")
            duser1["lastMonthPayedAmt"] = duser1["lastMonthPayedAmt"].fillna(0.0)
        elif idx==1:
            tmp["last_month_topay_num"] = tmp["pay_end_date"].apply(lambda x: 1 if (x- test_start).days/30.0 >=1 else (x- test_start).days/30.0)
            tmp["last_month_payed_amt"] = tmp["amt_per_plan"] * tmp["last_month_topay_num"]
            lastMonthPayedAmt= tmp.groupby("uid")["last_month_payed_amt"].sum().rename("lastMonthPayedAmt").reset_index()
            duser2 = duser2.merge(lastMonthPayedAmt, on="uid", how="left")
            duser2["lastMonthPayedAmt"] = duser2["lastMonthPayedAmt"].fillna(0.0)
    return duser1, duser2

def getPast3MonthLoanFeatures(df, duser1, duser2):
    valid_mask = df.month.isin([8,  9, 10])
    test_mask = df.month.isin([9, 10, 11])
    window_size = 92
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask]
        #平均每月贷款金额
        month_loanAmt = tmp.groupby(["uid","month"])['loan_amount'].sum().rename("monthLoanAmt").reset_index().groupby(["uid"])["monthLoanAmt"].mean().rename("monthAvgLoanAmt").reset_index()
        #month_loanAmt = tmp.groupby(["uid","month"])['loan_amount'].sum().rename("monthLoanAmt").reset_index().groupby(["uid"])["monthLoanAmt"].agg(["count","mean","max","median","min","std"]).reset_index()
        #month_loanAmt.columns = ['uid']+ [i+ '_monthLoanAmt_'+str(window_size) for i in list(month_loanAmt.columns)[1:]]
        #贷款金额 loan_amount
        stat_loanAmt = tmp.groupby(["uid"])['loan_amount'].agg(['sum','count','mean','max','min']).reset_index()
        stat_loanAmt.columns=['uid']+ [i+ '_loanAmt_'+str(window_size) for i in list(stat_loanAmt.columns)[1:]]
        #贷款期数 plannum
        stat_loanPlanNum=tmp.groupby(["uid"])['plannum'].agg(['sum','mean','max','min']).reset_index()
        stat_loanPlanNum.columns=['uid']+ [i+ '_loanPlanNum_'+str(window_size) for i in list(stat_loanPlanNum.columns)[1:]]
        #每期贷款额 amt_per_plan
        stat_amtPerPlan=tmp.groupby(["uid"])['amt_per_plan'].agg(['sum','mean','max','min']).reset_index()
        stat_amtPerPlan.columns = ['uid']+ [i+ '_amtPerPlan_'+str(window_size) for i in list(stat_amtPerPlan.columns)[1:]]
        #贷款周期 loan_interval
        stat_loanInterval=tmp.groupby(["uid"])['loan_interval'].agg(['mean','median','max','min']).reset_index()
        stat_loanInterval.columns=['uid']+ [i+ '_loanInterval_'+str(window_size) for i in list(stat_loanInterval.columns)[1:]]
        loan3Month = month_loanAmt.merge(stat_loanAmt, on="uid", how="left").merge(stat_loanPlanNum, on="uid", how="left").merge(stat_amtPerPlan,  on="uid", how="left").merge(stat_loanInterval, on="uid", how="left").fillna(0.0)
        if idx == 0:
            duser1 = duser1.merge(loan3Month, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(loan3Month, how="left", on="uid")
    return duser1, duser2



def loanTimeBetweenActivetime(df,duser1, duser2):
    valid_mask = df.month.isin([8,  9, 10])
    test_mask = df.month.isin([8, 9, 10, 11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask]
        uid_nearest_loan = tmp[tmp.groupby(['uid'])['loan_time'].transform(max) == tmp['loan_time']][["uid","loan_time","loan_amount"]] #用户最近一次借贷的情况
        uid_first_loan = tmp[tmp.groupby(['uid'])['loan_time'].transform(min) == tmp['loan_time']][["uid","loan_time","loan_amount"]] #用户第一天借贷的情况
        uid_nearest_loan.columns = ["uid", "nearest_loan_time", "nearest_loan_amt"]
        uid_first_loan.columns = ["uid", "first_loan_time", "first_loan_amt"]
        uid_loan = uid_nearest_loan.merge(uid_first_loan, on="uid", how="left")
        if idx == 0:
            duser1 = duser1.merge(uid_loan, on="uid", how="left")
            duser1["first_loantime_active_days"] = (duser1["active_date"] - duser1["first_loan_time"]).apply(lambda x: x.days)  #第一次借贷距离用户激活的时间
            duser1["first_loan_amount_limit"] = duser1["first_loan_amt"]/duser1["limit"]            
            duser1["nearest_loantime_active_days"] = (duser1["active_date"] - duser1["nearest_loan_time"]).apply(lambda x: x.days)  #最近一次借贷距离用户激活的时间
            duser1["nearest_loan_amount_limit"] = duser1["nearest_loan_amt"]/duser1["limit"]
            duser1.drop(["nearest_loan_time", "first_loan_time"], axis=1, inplace=True)
        elif idx == 1:
            duser2 = duser2.merge(uid_loan, on="uid", how="left")
            duser2["first_loantime_active_days"] = (duser2["active_date"] - duser2["first_loan_time"]).apply(lambda x: x.days)  #第一次借贷距离用户激活的时间
            duser2["first_loan_amount_limit"] = duser2["first_loan_amt"]/duser2["limit"]
            duser2["nearest_loantime_active_days"] = (duser2["active_date"] - duser2["nearest_loan_time"]).apply(lambda x: x.days)  #最近一次借贷距离用户激活的时间
            duser2["nearest_loan_amount_limit"] = duser2["nearest_loan_amt"]/duser2["limit"]
            duser2.drop(["nearest_loan_time", "first_loan_time"], axis=1, inplace=True)
    return duser1, duser2


def getOrderClickRatio(df_click, df_order, duser1, duser2):
    click_valid_mask = df_click.month.isin([8,9,10])
    click_test_mask = df_click.month.isin([9,10,11])
    order_valid_mask = df_order.month.isin([8,9,10])
    order_test_mask = df_order.month.isin([9,10,11])
    uid_valid_clicks = df_click[click_valid_mask].groupby("uid")["click_time"].count().rename("total_clicks_3month").reset_index()
    uid_test_clicks = df_click[click_test_mask].groupby("uid")["click_time"].count().rename("total_clicks_3month").reset_index()
    uid_valid_orders = df_order[order_valid_mask].groupby("uid")["buy_time"].count().rename("total_order_3month").reset_index()
    uid_test_orders = df_order[order_test_mask].groupby("uid")["buy_time"].count().rename("total_order_3month").reset_index()
    uid_valid_click_order = uid_valid_clicks.merge(uid_valid_orders, on="uid", how="left")
    uid_test_click_order = uid_test_clicks.merge(uid_test_orders, on="uid", how="left")
    uid_valid_click_order["click_order_ratio"] = uid_valid_click_order["total_clicks_3month"]/ (uid_valid_click_order["total_order_3month"] + 1)
    uid_test_click_order["click_order_ratio"] = uid_test_click_order["total_clicks_3month"]/ (uid_test_click_order["total_order_3month"] + 1)
    duser1 = duser1.merge(uid_valid_click_order, on="uid", how="left")
    duser2 = duser2.merge(uid_valid_click_order, on="uid", how="left")
    return duser1, duser2


def getNearest2LoanInterval(df_loan, duser1, duser2):
    valid_mask = df_loan.month.isin([8,  9, 10])
    test_mask = df_loan.month.isin([8, 9, 10, 11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df_loan[mask]
        nearestLastLoanInterval = tmp[tmp.groupby(['uid'])['loan_time'].transform(max) == tmp['loan_time']][["uid","loan_interval"]]  #用户最近一次借贷的前一次借贷间隔
        nearestLastLoanInterval.columns = ["uid","nearestLastLoanInterval"]
        if idx == 0:
            duser1 = duser1.merge(nearestLastLoanInterval, on ="uid", how="left")
        elif idx == 1:
            duser2 = duser2.merge(nearestLastLoanInterval, on ="uid", how="left")
    return duser1, duser2

##用户折扣率，提升很小
def userDiscountRatio(df_order, duser1, duser2):
    valid_mask = df_order.month.isin([8,9,10])
    test_mask = df_order.month.isin([9,10,11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df_order[mask].reset_index(drop=True)
        orderAmtDiscount = tmp.groupby("uid")["order_amt","discount"].sum().reset_index()
        orderAmtDiscount.columns = ["uid", "total_order_amt", "total_discount_amt"]
        orderAmtDiscount["discount_ratio"] = 1 - orderAmtDiscount["total_discount_amt"] /orderAmtDiscount["total_order_amt"]
        if idx == 0:
            duser1 = duser1.merge(orderAmtDiscount, on="uid", how="left")
            duser1["discount_ratio"] = duser1["discount_ratio"].fillna(1.0)
        elif idx ==1:
            duser2 = duser2.merge(orderAmtDiscount, on="uid", how="left")
            duser2["discount_ratio"] = duser2["discount_ratio"].fillna(1.0)
    return duser1, duser2


def getMonthLoanShiftDiff(df_loan, duser1, duser2):
    valid_mask = df_loan.month.isin([8,  9, 10])
    test_mask = df_loan.month.isin([8, 9, 10, 11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        uidMonthLoan = df_loan[mask].groupby(["uid","month"])["loan_amount"].sum().rename("month_loan_amt").reset_index()
        uidMonthLoan = uidMonthLoan.pivot(index='uid', columns='month', values='month_loan_amt').fillna(0)
        uidMonthLoan = uidMonthLoan.stack().reset_index()
        uidMonthLoan.columns =["uid","month","month_loan_amt"]
        uidMonthLoan = uidMonthLoan.groupby(["uid"]).apply(lambda x: x.sort_values(["month"], ascending=True)).reset_index(drop=True)
        uidMonthLoan["monthLoanAmtDiff"] = uidMonthLoan.groupby("uid")["month_loan_amt"].apply(lambda x: x - x.shift(1))
        uidMonthLoan["monthLoanAmtDiff2"] = uidMonthLoan.groupby("uid")["monthLoanAmtDiff"].apply(lambda x: x - x.shift(1))
        if idx == 0:
            duser1 = duser1.merge(uidMonthLoan[uidMonthLoan.month == 10][["uid", "monthLoanAmtDiff","monthLoanAmtDiff2"]], on ="uid", how="left")
        elif idx == 1:   
            duser2 = duser2.merge(uidMonthLoan[uidMonthLoan.month == 11][["uid", "monthLoanAmtDiff","monthLoanAmtDiff2"]], on ="uid", how="left")
    return duser1, duser2


def getLoanShiftDiff(df_loan, duser1, duser2):
    valid_mask = df_loan.month.isin([8,  9, 10])
    test_mask = df_loan.month.isin([8, 9, 10, 11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df_loan[mask].reset_index(drop=True)
        tmp["loanAmtDiff"] = tmp.groupby("uid")["loan_amount"].apply(lambda x: x - x.shift(1))
        tmp["loanAmtDiff2"] = tmp.groupby("uid")["loanAmtDiff"].apply(lambda x: x - x.shift(1))
        maxtime_idx = tmp.groupby(['uid'])['loan_time'].transform(max) == tmp['loan_time']  #用户最近一天贷款的情况
        tmp = tmp[maxtime_idx]
        if idx == 0:
            duser1 = duser1.merge(tmp[["uid", "loanAmtDiff","loanAmtDiff2"]], on ="uid", how="left")
        elif idx == 1:
            duser2 = duser2.merge(tmp[["uid", "loanAmtDiff","loanAmtDiff2"]], on ="uid", how="left")
    return duser1, duser2

def getMonthOrderShiftDiff(df_order, duser1, duser2):
    valid_mask = df_order.month.isin([8,  9, 10])
    test_mask = df_order.month.isin([8, 9, 10, 11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        uidMonthLoan = df_order[mask].groupby(["uid","month"])["order_amt"].sum().rename("month_order_amt").reset_index()
        uidMonthLoan = uidMonthLoan.pivot(index='uid', columns='month', values='month_order_amt').fillna(0)
        uidMonthLoan = uidMonthLoan.stack().reset_index()
        uidMonthLoan.columns =["uid","month","month_order_amt"]
        uidMonthLoan = uidMonthLoan.groupby(["uid"]).apply(lambda x: x.sort_values(["month"], ascending=True)).reset_index(drop=True)
        uidMonthLoan["monthOrderAmtDiff"] = uidMonthLoan.groupby("uid")["month_order_amt"].apply(lambda x: x - x.shift(1))
        uidMonthLoan["monthOrderAmtDiff2"] = uidMonthLoan.groupby("uid")["monthOrderAmtDiff"].apply(lambda x: x - x.shift(1))
        if idx == 0:
            duser1 = duser1.merge(uidMonthLoan[uidMonthLoan.month == 10][["uid", "monthOrderAmtDiff","monthOrderAmtDiff2"]], on ="uid", how="left")
        elif idx == 1:   
            duser2 = duser2.merge(uidMonthLoan[uidMonthLoan.month == 11][["uid", "monthOrderAmtDiff","monthOrderAmtDiff2"]], on ="uid", how="left")
    return duser1, duser2

def getLoanCntShiftDiff(df_loan, duser1, duser2):
    valid_mask = df_loan.month.isin([8,  9, 10])
    test_mask = df_loan.month.isin([8, 9, 10, 11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df_loan[mask].reset_index(drop=True)
        uidMonthLoan = tmp.groupby(["uid","month"])["loan_time"].count().rename("month_loan_cnt").reset_index()
        uidMonthLoan = uidMonthLoan.pivot(index='uid', columns='month', values='month_loan_cnt').fillna(0)
        uidMonthLoan = uidMonthLoan.stack().reset_index()
        uidMonthLoan.columns =["uid","month","month_loan_cnt"]
        uidMonthLoan = uidMonthLoan.groupby(["uid"]).apply(lambda x: x.sort_values(["month"], ascending=True)).reset_index(drop=True)
        uidMonthLoan["loanCntDiff"] = uidMonthLoan.groupby(["uid"])["month_loan_cnt"].apply(lambda x: x - x.shift(1))
        uidMonthLoan["loanCntDiff2"] = uidMonthLoan.groupby(["uid"])["loanCntDiff"].apply(lambda x: x - x.shift(1))
        if idx == 0:
            duser1 = duser1.merge(uidMonthLoan.loc[uidMonthLoan.month==10, ["uid", "loanCntDiff","loanCntDiff2"]], on ="uid", how="left")
        elif idx == 1:
            duser2 = duser2.merge(uidMonthLoan.loc[uidMonthLoan.month==11, ["uid", "loanCntDiff","loanCntDiff2"]], on ="uid", how="left")
    return duser1, duser2    

#math.ceil(34600/1000.0) * 1000
def updateLimit(duser1, duser2):
    loan_validmask = t_loan.month < 11
    loan_testmask = t_loan.month < 12
    for idx, mask in enumerate([loan_validmask, loan_testmask]):
        uid_day_loan = t_loan[mask].groupby(["uid","date"])["loan_amount"].sum().rename("new_limit").reset_index()
        uid_newlimit = uid_day_loan.groupby("uid")["new_limit"].max().rename("new_limit").reset_index()
        uid_newlimit = uid_newlimit.merge(t_user[["uid","limit"]], on="uid", how="left")
        updateLimit = uid_newlimit[uid_newlimit["limit"] < uid_newlimit["new_limit"]][["uid","new_limit"]]
        if idx ==0:
            duser1 = duser1.merge(updateLimit, on="uid", how="left")
            duser1["limit"] = duser1.apply(lambda x: x["new_limit"] if x["limit"] < x["new_limit"] else x["limit"], axis=1)
            duser1.drop("new_limit",axis=1,inplace=True)
        elif idx == 1:
            duser2 = duser2.merge(updateLimit, on="uid", how="left")
            duser2["limit"] = duser2.apply(lambda x: x["new_limit"] if x["limit"] < x["new_limit"] else x["limit"], axis=1)
            duser2.drop("new_limit",axis=1,inplace=True)
    return duser1, duser2



def updateLimit2(duser1, duser2):
    loan_validmask = (t_loan.month < 11) 
    loan_testmask = (t_loan.month < 12)
    for idx, mask in enumerate([loan_validmask, loan_testmask]):
        uid_newlimit = t_loan[mask & (t_loan.plannum>6)].groupby(["uid","month"])["loan_amount"].sum().rename("loan12amt").reset_index().groupby("uid")["loan12amt"].max().rename("new_limit").reset_index()
        uid_day_loan = t_loan[mask].groupby(["uid","date"])["loan_amount"].sum().rename("new_limit").reset_index().groupby("uid")["new_limit"].max().rename("new_limit").reset_index()
        uid_newlimit_all = pd.concat([uid_newlimit,uid_day_loan]).groupby("uid")["new_limit"].max().rename("new_limit").reset_index()
        uid_newlimit = uid_newlimit_all.merge(t_user[["uid","limit"]], on="uid", how="left")
        updateLimit = uid_newlimit[uid_newlimit["limit"] < uid_newlimit["new_limit"]][["uid","new_limit"]]
        if idx ==0:
            duser1 = duser1.merge(updateLimit, on="uid", how="left")
            duser1["limit"] = duser1.apply(lambda x: x["new_limit"] if x["limit"] < x["new_limit"] else x["limit"], axis=1)
            duser1.drop("new_limit",axis=1,inplace=True)
        elif idx == 1:
            duser2 = duser2.merge(updateLimit, on="uid", how="left")
            duser2["limit"] = duser2.apply(lambda x: x["new_limit"] if x["limit"] < x["new_limit"] else x["limit"], axis=1)
            duser2.drop("new_limit",axis=1,inplace=True)
    return duser1, duser2




def getSexAgeLimt(dt_user, duser1):
    ori_limit = dt_user[["uid","limit"]]
    ori_limit.columns = ["uid", "ori_limit"]
    duser1 = duser1.merge(ori_limit, on="uid", how="left")
    duser1["limit_increase"] = duser1["limit"]/duser1["ori_limit"]
    duser1["sex_age_limit"] = duser1["sex"].astype(str) +  duser1["age"].astype(str) + duser1["ori_limit"].astype(str)
    duser1["sex_age_limit"] = duser1["sex_age_limit"].astype('category')
    duser1['sex_age_limit'].cat.categories= np.arange(1,duser1["sex_age_limit"].nunique()+1)
    duser1["sex_age_limit"] = duser1["sex_age_limit"].astype(int)
    return duser1

def getAmtBeforeRatio(df, column ,duser1, duser2):
    valid_mask = df.month.isin([10,9,8])
    test_mask = df.month.isin([11,10,9,8])
    for idx, mask in enumerate([valid_mask, test_mask]):
        uid_months = df[mask].groupby(["uid","month"])[column].agg(["count","sum"]).reset_index()
        uid_months.rename({'count': column + '_cnt', 'sum': column + '_sum' }, axis='columns',inplace=True)
        if idx == 0:
            uid_valid = uid_months[uid_months.month==10].reset_index(drop=True)
            mean_before_month = uid_months[uid_months.month < 10].groupby("uid")[column + '_sum'].mean().rename("mean_before_month").reset_index()
            uid_valid = uid_valid.merge(mean_before_month, how="left", on="uid")
            uid_valid[column + '_sum_before_ratio'] = uid_valid[column + '_sum']/ (uid_valid['mean_before_month'] + 1)
        elif idx == 1:
            uid_valid = uid_months[uid_months.month==11].reset_index(drop=True)
            mean_before_month = uid_months[uid_months.month < 11].groupby("uid")[column + '_sum'].mean().rename("mean_before_month").reset_index()
            uid_valid = uid_valid.merge(mean_before_month, how="left", on="uid")
            uid_valid[column + '_sum_before_ratio'] = uid_valid[column + '_sum']/ (uid_valid['mean_before_month'] + 1)
        if idx == 0:
            duser1 = duser1.merge(uid_valid[["uid",column + '_sum_before_ratio']], how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(uid_valid[["uid",column + '_sum_before_ratio']], how="left", on="uid")
    return duser1, duser2

def getActionDays(df, column ,duser1, duser2, window_size, pref):
    valid_mask, test_mask = get_windows_mask(df, column, window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        uid_months = df[mask].groupby(["uid"])[column].nunique().rename(pref + "_actionDays").reset_index()
        if idx == 0:
            duser1 = duser1.merge(uid_months, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(uid_months, how="left", on="uid")
    return duser1, duser2


def currentMinDebtAmt(df, duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(df, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        if idx == 0:
            pay_date_mask = df["pay_end_date"] > pd.Timestamp(valid_end_date)
            tmp = df[mask & pay_date_mask].reset_index(drop=True)
            tmp["payed_num"] = (pd.Timestamp(valid_end_date) - tmp["loan_time"]).apply(lambda x: math.ceil(x.days/30.0))
            tmp["debtAmt"] = tmp["loan_amount"] - tmp["amt_per_plan"] * tmp["payed_num"]
            current_debtAmt = tmp.groupby("uid")["debtAmt"].sum().rename("current_MindebtAmt_" + str(window_size)).reset_index()
            duser1 = duser1.merge(current_debtAmt, on="uid", how="left")
            duser1["current_MindebtAmt_" + str(window_size)] = duser1["current_MindebtAmt_" + str(window_size)].fillna(0.0)
        elif idx == 1:
            pay_date_mask = df["pay_end_date"] > pd.Timestamp(valid_end_date)
            tmp = df[mask & pay_date_mask].reset_index(drop=True)
            tmp["payed_num"] = (pd.Timestamp(test_end_date) - tmp["loan_time"]).apply(lambda x: math.ceil(x.days/30.0))
            tmp["debtAmt"] = tmp["loan_amount"] - tmp["amt_per_plan"] * tmp["payed_num"]
            current_debtAmt = tmp.groupby("uid")["debtAmt"].sum().rename("current_MindebtAmt_" + str(window_size)).reset_index()
            duser2 = duser2.merge(current_debtAmt, on="uid", how="left")
            duser2["current_MindebtAmt_" + str(window_size)] = duser2["current_MindebtAmt_" + str(window_size)].fillna(0.0)
    return duser1, duser2


def currentMaxDebtAmt(df, duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(df, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        if idx == 0:
            pay_date_mask = df["pay_end_date"] > pd.Timestamp(valid_end_date)
            tmp = df[mask & pay_date_mask].reset_index(drop=True)
            tmp["payed_num"] = (pd.Timestamp(valid_end_date) - tmp["loan_time"]).apply(lambda x: math.floor(x.days/30.0))
            tmp["debtAmt"] = tmp["loan_amount"] - tmp["amt_per_plan"] * tmp["payed_num"]
            current_debtAmt = tmp.groupby("uid")["debtAmt"].sum().rename("current_MaxdebtAmt_" + str(window_size)).reset_index()
            duser1 = duser1.merge(current_debtAmt, on="uid", how="left")
            duser1["current_MaxdebtAmt_" + str(window_size)] = duser1["current_MaxdebtAmt_" + str(window_size)].fillna(0.0)
        elif idx == 1:
            pay_date_mask = df["pay_end_date"] > pd.Timestamp(valid_end_date)
            tmp = df[mask & pay_date_mask].reset_index(drop=True)
            tmp["payed_num"] = (pd.Timestamp(test_end_date) - tmp["loan_time"]).apply(lambda x: math.floor(x.days/30.0))
            tmp["debtAmt"] = tmp["loan_amount"] - tmp["amt_per_plan"] * tmp["payed_num"]
            current_debtAmt = tmp.groupby("uid")["debtAmt"].sum().rename("current_MaxdebtAmt_" + str(window_size)).reset_index()
            duser2 = duser2.merge(current_debtAmt, on="uid", how="left")
            duser2["current_MaxdebtAmt_" + str(window_size)] = duser2["current_MaxdebtAmt_" + str(window_size)].fillna(0.0)
    return duser1, duser2


##下个月需要还多少钱
def net2PayAmt(df, duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(df, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        if idx == 0:
            pay_date_mask = df["pay_end_date"] > pd.Timestamp(valid_end_date) + timedelta(days=30)
            tmp = df[mask & pay_date_mask]
            net2PayAmt = tmp.groupby("uid")["amt_per_plan"].agg(["sum"]).reset_index()
            net2PayAmt.columns = ["uid", "net2PayAmt"]
            duser1 = duser1.merge(net2PayAmt, on="uid", how="left")
            duser1["net2PayAmt"] = duser1["net2PayAmt"].fillna(0.0)
        elif idx == 1:
            pay_date_mask = df["pay_end_date"] > pd.Timestamp(test_end_date) + timedelta(days=30)
            tmp = df[mask & pay_date_mask]
            net2PayAmt = tmp.groupby("uid")["amt_per_plan"].agg(["sum"]).reset_index()
            net2PayAmt.columns = ["uid", "net2PayAmt"]
            duser2 = duser2.merge(net2PayAmt, on="uid", how="left")
            duser2["net2PayAmt"] = duser2["net2PayAmt"].fillna(0.0)
    return duser1, duser2

def orderAmtStaus(df, duser, tuser, window_size):
    valid_mask, test_mask = get_windows_mask(df, "buy_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask].reset_index(drop=True)
        if idx == 0:
            tmp["buy_time_interval"] = tmp["buy_time"]- pd.Timestamp(valid_end_date)
            tmp["buy_time_interval"] = tmp["buy_time_interval"].apply(lambda x: x.days)
            uid_order_his= tmp.groupby(["uid"])["buy_time_interval"].mean().rename("buy_time_interval_mean").reset_index()
            duser = duser.merge(uid_order_his, how="left", on = 'uid')
        elif idx == 1:
            tmp["buy_time_interval"] = tmp["buy_time"]- pd.Timestamp(test_end_date)
            tmp["buy_time_interval"] = tmp["buy_time_interval"].apply(lambda x: x.days)
            uid_order_his= tmp.groupby(["uid"])["buy_time_interval"].mean().rename("buy_time_interval_mean").reset_index()
            tuser = tuser.merge(uid_order_his, how="left", on = 'uid')
    return duser, tuser

def gen_fixed_tw_features_for_click_PidParamUnFold(df,duser1,duser2,col,window_size):
    valid_mask, test_mask = get_windows_mask(df, col, window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask].reset_index(drop=True)
        tmp['pidParam']=pd.Series([str(i)+"_"+str(j) for i,j in zip(list(tmp.pid),list(tmp.param))],index=tmp.index)
        uid_pidParam_clicks = tmp.groupby(["uid","pidParam"]).click_time.count().reset_index()
        uid_pidParam_clicks.columns = ["uid","pidParam", "pidParam_clicks"]
        uid_pidParam_clicks["pidParam"]  =  uid_pidParam_clicks['pidParam'].astype(str) + "_pidParamcliks_" + str(window_size) + "d"
        uid_pidParam_clicks = uid_pidParam_clicks.pivot(index='uid', columns='pidParam', values='pidParam_clicks').reset_index().fillna(0)
        if idx == 0:
            duser1 = duser1.merge(uid_pidParam_clicks, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(uid_pidParam_clicks, how="left", on="uid")           
        uid_pidParam_clicks = tmp.groupby(["uid","pidParam"]).click_time.count().reset_index()
        uid_pidParam_clicks.columns = ["uid","pidParam", "pidParam_clicks"]        
        uid_clicks = tmp.groupby(["uid"]).click_time.count().rename('clicks').reset_index()       
        uid_pidParam_clicks = uid_pidParam_clicks.merge(uid_clicks,how="left",on="uid")
        uid_pidParam_clicks["clicks_pidParam_ratio"] = uid_pidParam_clicks["pidParam_clicks"]/uid_pidParam_clicks["clicks"]
        uid_pidParam_clicks["pidParam"]  =  uid_pidParam_clicks['pidParam'].astype(str) + "_pidParamcliks_ratio_" + str(window_size) + "d"
        uid_pidParam_clicks = uid_pidParam_clicks.pivot(index='uid', columns='pidParam', values='clicks_pidParam_ratio').reset_index().fillna(0)
        if idx == 0:
            duser1 = duser1.merge(uid_pidParam_clicks, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(uid_pidParam_clicks, how="left", on="uid")   
    return duser1, duser2


def getLoanFeaturesWinds(df, duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(df, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask]
        #平均每月贷款金额
        month_loanAmt = tmp.groupby(["uid"])['loan_amount'].sum().rename("monthLoanAmt").reset_index()
        #贷款周期 loan_interval
        stat_loanInterval=tmp.groupby(["uid"])['loan_interval'].agg(['mean','median','max','min']).reset_index()
        stat_loanInterval.columns=['uid']+ [i+ '_loanInterval_'+str(window_size) for i in list(stat_loanInterval.columns)[1:]]
        loan3Month = month_loanAmt.merge(stat_loanInterval,  on="uid", how="left")
        if idx == 0:
            duser1 = duser1.merge(loan3Month, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(loan3Month, how="left", on="uid")
    return duser1, duser2


def gen_fixed_tw_features_for_loan_click_PidParamUnfold(df1,df2,duser1,duser2,col1,col2,window_size):
    valid_mask_loan, test_mask_loan = get_windows_mask(df1, col1, window_size)
    valid_mask_click, test_mask_click = get_windows_mask(df2, col2, window_size)
    for idx, mask in enumerate([valid_mask_loan, test_mask_loan]):        
        tmp_loan = df1[mask].reset_index(drop=True)
        if idx==0:
            mask_click=valid_mask_click
        elif idx==1:
            mask_click=test_mask_click           
        tmp_click= df2[mask_click].reset_index(drop=True)
        tmp_click['pidParam']=pd.Series([str(i)+"_"+str(j) for i,j in zip(list(tmp_click.pid),list(tmp_click.param))],index=tmp_click.index)
        uid_cnt_pidparam_by_day=tmp_click.groupby(['uid','date','pidParam']).click_time.count().rename('cnt_PidParam_in_loan').reset_index()
        uid_cnt_loan_by_day=tmp_loan.groupby(['uid','date']).loan_time.count().rename('cnt_loan').reset_index()
        del uid_cnt_loan_by_day['cnt_loan']
        uid_loan_click=uid_cnt_loan_by_day.merge(uid_cnt_pidparam_by_day,how='left',on=['uid','date'])
        uid_loan_click_tw=uid_loan_click.groupby(['uid','pidParam']).cnt_PidParam_in_loan.sum().rename('cnt_PidParam').reset_index()
        uid_loan_click_total=uid_loan_click.groupby('uid').cnt_PidParam_in_loan.sum().rename('cnt_total').reset_index()
        uid_loan_click_tw=uid_loan_click_tw.merge(uid_loan_click_total,how='left',on=['uid'])   
        uid_loan_click_tw['ratio_pidParam']=uid_loan_click_tw.cnt_PidParam/uid_loan_click_tw.cnt_total
        uid_loan_click_tw['cnt_pidParam_in_loan']=uid_loan_click_tw['pidParam'].astype(str)+"_pidParamcliks_in_loan_" + str(window_size) + "d"
        uid_loan_click_tw['ratio_pidParam_in_loan']=uid_loan_click_tw['pidParam'].astype(str)+"_pidParamRatio_in_loan_" + str(window_size) + "d"
        uid_loan_click_cnt=uid_loan_click_tw.pivot(index='uid', columns='cnt_pidParam_in_loan', values='cnt_PidParam').reset_index().fillna(0)
        uid_loan_click_ratio=uid_loan_click_tw.pivot(index='uid', columns='ratio_pidParam_in_loan', values='ratio_pidParam').reset_index().fillna(0)
        if idx == 0:
            duser1 = duser1.merge(uid_loan_click_cnt, how="left", on="uid")
            duser1 = duser1.merge(uid_loan_click_ratio, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(uid_loan_click_cnt, how="left", on="uid")
            duser2 = duser2.merge(uid_loan_click_ratio, how="left", on="uid")   
    return duser1, duser2

def gen_fixed_tw_features_for_notloan_click_PidParamUnfold(df1,df2,duser1,duser2,col1,col2,window_size):
    valid_mask_loan, test_mask_loan = get_windows_mask(df1, col1, window_size)
    valid_mask_click, test_mask_click = get_windows_mask(df2, col2, window_size)
    for idx, mask in enumerate([valid_mask_loan, test_mask_loan]):        
        tmp_loan = df1[mask].reset_index(drop=True)
        if idx==0:
            mask_click=valid_mask_click
        elif idx==1:
            mask_click=test_mask_click           
        tmp_click= df2[mask_click].reset_index(drop=True)
        tmp_click['pidParam']=pd.Series([str(i)+"_"+str(j) for i,j in zip(list(tmp_click.pid),list(tmp_click.param))],index=tmp_click.index)
        uid_cnt_pidparam_by_day=tmp_click.groupby(['uid','date','pidParam']).click_time.count().rename('cnt_PidParam_notin_loan').reset_index()
        uid_cnt_loan_by_day=tmp_loan.groupby(['uid','date']).loan_time.count().rename('cnt_loan').reset_index()
        uid_loan_click= uid_cnt_pidparam_by_day.merge(uid_cnt_loan_by_day,how='left',on=['uid','date'])
        uid_loan_click=uid_loan_click[uid_loan_click.cnt_loan.isnull()]
        uid_loan_click_tw=uid_loan_click.groupby(['uid','pidParam']).cnt_PidParam_notin_loan.sum().rename('cnt_PidParam').reset_index()
        uid_loan_click_total=uid_loan_click.groupby('uid').cnt_PidParam_notin_loan.sum().rename('cnt_total').reset_index()
        uid_loan_click_tw=uid_loan_click_tw.merge(uid_loan_click_total,how='left',on=['uid'])   
        uid_loan_click_tw['ratio_pidParam']=uid_loan_click_tw.cnt_PidParam/uid_loan_click_tw.cnt_total
        uid_loan_click_tw['cnt_pidParam_notin_loan']=uid_loan_click_tw['pidParam'].astype(str)+"_pidParamcliks_notin_loan_" + str(window_size) + "d"
        uid_loan_click_tw['ratio_pidParam_notin_loan']=uid_loan_click_tw['pidParam'].astype(str)+"_pidParamRatio_notin_loan_" + str(window_size) + "d"
        uid_loan_click_cnt=uid_loan_click_tw.pivot(index='uid', columns='cnt_pidParam_notin_loan', values='cnt_PidParam').reset_index().fillna(0)
        uid_loan_click_ratio=uid_loan_click_tw.pivot(index='uid', columns='ratio_pidParam_notin_loan', values='ratio_pidParam').reset_index().fillna(0)
        if idx == 0:
            duser1 = duser1.merge(uid_loan_click_cnt, how="left", on="uid")
            duser1 = duser1.merge(uid_loan_click_ratio, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(uid_loan_click_cnt, how="left", on="uid")
            duser2 = duser2.merge(uid_loan_click_ratio, how="left", on="uid")   
    return duser1, duser2
    
def getCatePivotAmtCnt(df, duser1, duser2):
    valid_mask = df.month.isin([8, 9,10])
    test_mask = df.month.isin([9, 10,11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask]
        uid_months = tmp.groupby(["uid","cate_id"])["order_amt"].agg(["count","sum"]).reset_index()
        uid_months.rename({'count': 'cate_id_cnt', 'sum': 'cate_id_sum' }, axis='columns',inplace=True)
        if idx == 0:
            uid_months["cate_id"]  = "cate_id_" + uid_months['cate_id'].astype(str)
        elif idx == 1:
            uid_months["cate_id"]  = "cate_id_" + uid_months['cate_id'].astype(str)
        uid_months = uid_months.pivot(index='uid', columns='cate_id').reset_index().fillna(0)
        new_list = ["uid"]
        for words in uid_months.columns.get_values():
            if "uid" in words:
                continue
            new_list.append('_'.join(words))
        uid_months.columns =  new_list
        if idx == 0:
            duser1 = duser1.merge(uid_months, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(uid_months, how="left", on="uid")
    return duser1, duser2   

def uidOrderAmtCntWinds(df, duser, tuser, window_size):
    valid_mask, test_mask = get_windows_mask(df, "buy_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask]
        order_status = tmp.groupby(["uid"])["order_amt"].agg(["count","sum"]).reset_index()
        order_status.columns = ['uid', 'order_cnt_'+str(window_size), "order_amt_" + str(window_size) ]
        if idx == 0:
            duser = duser.merge(order_status, how="left", on = 'uid')
        elif idx == 1:
            tuser = tuser.merge(order_status, how="left", on = 'uid')
        return duser, tuser

def loanClickBehiviorSeries(duser1,duser2,window_size):
    valid_mask_loan, test_mask_loan = get_windows_mask(t_loan, "loan_time", window_size)
    valid_mask_click, test_mask_click = get_windows_mask(t_click, "click_time", window_size)
    for idx, mask in enumerate([valid_mask_loan, test_mask_loan]):
        tmp_loan = t_loan.loc[mask,['uid','date']].rename(columns={'date':'behavior_time'})
        tmp_loan['behavior'] = 0
        if idx == 0:
            mask_click = valid_mask_click
        elif idx == 1:
            mask_click = test_mask_click 
        tmp_click=t_click.loc[mask_click,['uid','date']].rename(columns={'date':'behavior_time'})
        tmp_click['behavior']=1
        tmp=pd.concat([tmp_loan.drop_duplicates(),tmp_click.drop_duplicates()]).groupby(["uid"]).apply(lambda x: x.sort_values(["behavior_time"], ascending=True)).reset_index(drop=True)
        tmp1 = tmp.groupby("uid")["behavior"].apply(lambda x:list(x)).rename("clickLoanBehavior").reset_index()
        tmp1["clickLoanBehaviorCode"] = tmp1["clickLoanBehavior"].apply(lambda x: ''.join(map(str, x)))
        #tmp1["clickLoanBehaviorCode1"] = tmp1["clickLoanBehaviorCode"].apply(lambda x:len(x))
        tmp1["clickLoanBehaviorCode"] = tmp1["clickLoanBehaviorCode"].astype('category')
        tmp1['clickLoanBehaviorCode'].cat.categories=range(tmp1["clickLoanBehaviorCode"].nunique())  #4563
        tmp1["clickLoanBehaviorCode"] = tmp1["clickLoanBehaviorCode"].astype(int)
        if idx == 0:
            duser1 = duser1.merge(tmp1[["uid","clickLoanBehaviorCode"]], on="uid", how="left")
        elif idx == 1:
            duser2 = duser2.merge(tmp1[["uid","clickLoanBehaviorCode"]], on="uid", how="left")
    return duser1, duser2

def loanClickBehiviorSeries1(duser1,duser2,window_size):
    valid_mask_loan, test_mask_loan = get_windows_mask(t_loan, "loan_time", window_size)
    valid_mask_click, test_mask_click = get_windows_mask(t_click, "click_time", window_size)
    for idx, mask in enumerate([valid_mask_loan, test_mask_loan]):
        tmp_loan = t_loan.loc[mask,['uid','date']].rename(columns={'date':'behavior_time'})
        tmp_loan['behavior'] = 0
        if idx == 0:
            mask_click = valid_mask_click
        elif idx == 1:
            mask_click = test_mask_click 
        tmp_click=t_click.loc[mask_click,['uid','date']].rename(columns={'date':'behavior_time'})
        tmp_click['behavior']=1
        tmp=pd.concat([tmp_loan.drop_duplicates(),tmp_click.drop_duplicates()]).groupby(["uid"]).apply(lambda x: x.sort_values(["behavior_time"], ascending=True)).reset_index(drop=True)
        tmp["pre_behavior"] = tmp.groupby("uid")["behavior"].apply(lambda x: x.shift(1))
        tmp = tmp[tmp["behavior"] != tmp["pre_behavior"]]
        tmp1 = tmp.groupby("uid")["behavior"].apply(lambda x:list(x)).rename("clickLoanBehavior").reset_index()
        tmp1["clickLoanBehaviorCode"] = tmp1["clickLoanBehavior"].apply(lambda x: ''.join(map(str, x)))
        #tmp1["clickLoanBehaviorCode1"] = tmp1["clickLoanBehaviorCode"].apply(lambda x:len(x))
        tmp1["clickLoanBehaviorCode"] = tmp1["clickLoanBehaviorCode"].astype('category')
        tmp1['clickLoanBehaviorCode'].cat.categories=range(tmp1["clickLoanBehaviorCode"].nunique())  #4563
        tmp1["clickLoanBehaviorCode"] = tmp1["clickLoanBehaviorCode"].astype(int)
        if idx == 0:
            duser1 = duser1.merge(tmp1[["uid","clickLoanBehaviorCode"]], on="uid", how="left")
        elif idx == 1:
            duser2 = duser2.merge(tmp1[["uid","clickLoanBehaviorCode"]], on="uid", how="left")
    return duser1, duser2

#将购买表中cateId展开（cnt amt ratio_cnt ratio_amt）
def gen_fixed_tw_features_for_order_cateIdUnFold(df,duser1,duser2, col,window_size):
    valid_mask, test_mask = get_windows_mask(df, col, window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask]
        uid_orders_cnts = tmp.groupby(["uid","cate_id"]).buy_time.count().rename('cnt_cateId').reset_index()
        uid_orders_amt = tmp.groupby(["uid","cate_id"]).order_amt.sum().rename('sum_cateId').reset_index()
        uid_orders_cnts_total=tmp.groupby(["uid"]).buy_time.count().rename('cnt_cateId_total').reset_index()
        uid_orders_amt_total=tmp.groupby(["uid"]).order_amt.sum().rename('sum_cateId_total').reset_index()
        uid_orders_cnts=uid_orders_cnts.merge(uid_orders_cnts_total,how='left',on='uid')
        uid_orders_amt=uid_orders_amt.merge(uid_orders_amt_total,how='left',on='uid')
        uid_orders_cnts['ratio_cnt_cateId']=uid_orders_cnts.cnt_cateId/uid_orders_cnts.cnt_cateId_total
        uid_orders_amt['ratio_amt_cateId']=uid_orders_amt.sum_cateId/uid_orders_amt.sum_cateId_total
        uid_orders_cnts['cate_id_cnt_fold']=uid_orders_cnts['cate_id'].astype(str)+"_cnt_" + str(window_size) + "d"
        uid_orders_cnts['cate_id_cnt_ratio_fold']=uid_orders_cnts['cate_id'].astype(str)+"_cnt_ratio_" + str(window_size) + "d"
        uid_orders_amt['cate_id_amt_fold']=uid_orders_amt['cate_id'].astype(str)+"_amt_" + str(window_size) + "d"
        uid_orders_amt['cate_id_amt_ratio_fold']=uid_orders_amt['cate_id'].astype(str)+"_amt_ratio_" + str(window_size) + "d"
        uid_orders_cnts_cnt=uid_orders_cnts.pivot(index='uid', columns='cate_id_cnt_fold', values='cnt_cateId').reset_index().fillna(0)
        uid_orders_cnts_ratio=uid_orders_cnts.pivot(index='uid', columns='cate_id_cnt_ratio_fold', values='ratio_cnt_cateId').reset_index().fillna(0)
        uid_orders_amt_amt=uid_orders_amt.pivot(index='uid', columns='cate_id_amt_fold', values='sum_cateId').reset_index().fillna(0)
        uid_orders_amt_ratio=uid_orders_amt.pivot(index='uid', columns='cate_id_amt_ratio_fold', values='ratio_amt_cateId').reset_index().fillna(0)
        d_feature=pd.DataFrame({'uid':t_user.uid},index=t_user.index)
        d_feature=d_feature.merge(uid_orders_cnts_cnt,how="left", on="uid").fillna(0)
        d_feature=d_feature.merge(uid_orders_amt_amt,how="left", on="uid").fillna(0)
        d_feature=d_feature.merge(uid_orders_cnts_ratio,how="left", on="uid")
        d_feature=d_feature.merge(uid_orders_amt_ratio,how="left", on="uid")        
        if idx == 0:
            duser1 = duser1.merge(d_feature, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(d_feature, how="left", on="uid")
    return duser1,duser2

def getTotalPidStaytime(df_click, duser1, duser2):
    valid_mask = df_click.month.isin([8,9,10])
    test_mask = df_click.month.isin([11,10,9])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df_click[mask].groupby(["uid"])["click_interval"].sum().rename("Staytime").reset_index()
        if idx == 0:
            duser1 = duser1.merge(tmp, on ="uid", how="left")
        elif idx == 1:   
            duser2 = duser2.merge(tmp, on ="uid", how="left")
    return duser1, duser2   

####非购买日期的点击参数展开
def gen_fixed_tw_features_for_notOrder_click_PidParamUnfold(df1,df2,duser1,duser2,col1,col2,window_size):
    valid_mask_order, test_mask_order = get_windows_mask(df1, col1, window_size)
    valid_mask_click, test_mask_click = get_windows_mask(df2, col2, window_size)
    for idx, mask in enumerate([valid_mask_order, test_mask_order]):        
        tmp_order = df1[mask].reset_index(drop=True)
        if idx==0:
            mask_click=valid_mask_click
        elif idx==1:
            mask_click=test_mask_click           
        tmp_click= df2[mask_click].reset_index(drop=True)
        tmp_click['pidParam']=pd.Series([str(i)+"_"+str(j) for i,j in zip(list(tmp_click.pid),list(tmp_click.param))],index=tmp_click.index)
        uid_cnt_pidparam_by_day=tmp_click.groupby(['uid','date','pidParam']).click_time.count().rename('cnt_PidParam_notin_order').reset_index()
        uid_cnt_order_by_day=tmp_order.groupby(['uid','buy_time']).buy_time.count().rename('cnt_order').reset_index()
        uid_cnt_order_by_day.columns = ["uid","date","cnt_order"]
        uid_order_click= uid_cnt_pidparam_by_day.merge(uid_cnt_order_by_day,how='left',on=['uid','date'])
        uid_order_click=uid_order_click[uid_order_click.cnt_order.isnull()]
        uid_order_click_tw=uid_order_click.groupby(['uid','pidParam']).cnt_PidParam_notin_order.sum().rename('cnt_PidParam').reset_index()
        uid_order_click_total=uid_order_click.groupby('uid').cnt_PidParam_notin_order.sum().rename('cnt_total').reset_index()
        uid_order_click_tw=uid_order_click_tw.merge(uid_order_click_total,how='left',on=['uid'])   
        uid_order_click_tw['ratio_pidParam']=uid_order_click_tw.cnt_PidParam/uid_order_click_tw.cnt_total
        uid_order_click_tw['cnt_pidParam_notin_order']=uid_order_click_tw['pidParam'].astype(str)+"_pidParamcliks_notin_order_" + str(window_size) + "d"
        uid_order_click_tw['ratio_pidParam_notin_order']=uid_order_click_tw['pidParam'].astype(str)+"_pidParamRatio_notin_order_" + str(window_size) + "d"
        uid_order_click_cnt=uid_order_click_tw.pivot(index='uid', columns='cnt_pidParam_notin_order', values='cnt_PidParam').reset_index().fillna(0)
        uid_order_click_ratio=uid_order_click_tw.pivot(index='uid', columns='ratio_pidParam_notin_order', values='ratio_pidParam').reset_index().fillna(0)
        if idx == 0:
            duser1 = duser1.merge(uid_order_click_cnt, how="left", on="uid")
            duser1 = duser1.merge(uid_order_click_ratio, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(uid_order_click_cnt, how="left", on="uid")
            duser2 = duser2.merge(uid_order_click_ratio, how="left", on="uid")  
    return duser1, duser2

def getLoanAmtWithinOrder(duser1, duser2, window_size):
    valid_mask_loan, test_mask_loan = get_windows_mask(t_loan, "loan_time", window_size)
    valid_mask_buy, test_mask_buy = get_windows_mask(t_order, "buy_time", window_size)
    for idx, mask in enumerate([valid_mask_loan, test_mask_loan]):        
        tmp_loan = t_loan[mask]
        loanAmtWinds = tmp_loan.groupby(["uid","date"])["loan_amount"].sum().rename('loanAmtIn').reset_index()
        if idx==0:
            tmp_order = t_order[valid_mask_buy]
        elif idx==1:
            tmp_order = t_order[test_mask_buy]
        orderAmtWinds = tmp_order.groupby(["uid","buy_time"])["order_amt"].sum().rename("orderAmtIn").reset_index()
        orderAmtWinds.columns = ["uid", "date", "orderAmtIn"]
        orderLoanAmtWinds = orderAmtWinds.merge(loanAmtWinds, how="inner", on=["uid","date"])
        loanAmtWinds= orderLoanAmtWinds.groupby("uid")['loanAmtIn'].sum().rename('loanAmtIn' + str(window_size)).reset_index()
        orderAmtWinds= orderLoanAmtWinds.groupby("uid")['orderAmtIn'].sum().rename('orderAmtIn' + str(window_size)).reset_index()
        orderLoanAmtWinds = loanAmtWinds.merge(orderAmtWinds, how="left", on="uid")
        orderLoanAmtWinds['loanOrderAmtRatioIn' + str(window_size)] = orderLoanAmtWinds['loanAmtIn' + str(window_size)] /orderLoanAmtWinds["orderAmtIn" + str(window_size)]
        if idx == 0:
            duser1 = duser1.merge(orderLoanAmtWinds, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(orderLoanAmtWinds, how="left", on="uid")
    return duser1, duser2

def orderCntDays(duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(t_order, "buy_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = t_order[mask].groupby("uid")["buy_time"].agg(["count","nunique"]).reset_index()
        tmp.columns = ["uid", "buyCnt" + str(window_size), "buyDays"+ str(window_size)]
        if idx == 0:
            duser1 = duser1.merge(tmp, on="uid", how="left")
        elif idx == 1:   
            duser2 = duser2.merge(tmp, on="uid", how="left")
    return duser1, duser2


def gen_fixed_tw_features_for_click_1m(df,col,duser1,duser2):
    valid_mask, test_mask = get_windows_mask(df, col, 90)
    for idx, mask in enumerate([valid_mask, test_mask]): 
        tmp=df[mask].reset_index(drop=True)
        if idx==0:
            tmp['cut_1m']=pd.cut(tmp.click_time,valid_cut_point,labels=labels,right=False,include_lowest=True)
            tmp=tmp[tmp.cut_1m.notnull()].reset_index(drop=True)
        elif idx==1:
            tmp['cut_1m']=pd.cut(tmp.click_time,test_cut_point,labels=labels,right=False,include_lowest=True)
            tmp=tmp[tmp.cut_1m.notnull()].reset_index(drop=True)
        stat_click_1m=tmp.groupby(['uid','cut_1m']).click_time.agg(['count']).reset_index()
        stat_click_1m.columns=['uid','cut_1m','click_count']
        cnt_total=stat_click_1m.groupby('uid').click_count.sum().rename('cnt_total').reset_index()
        stat_click_1m=stat_click_1m.merge(cnt_total,how='left',on='uid')
        stat_click_1m['ratio_cnt']=stat_click_1m.click_count/stat_click_1m.cnt_total
        stat_click_1m['count_1m']=stat_click_1m['cut_1m'].astype(str)+'_click_count'
        stat_click_1m['ratio_cnt_1m']=stat_click_1m['cut_1m'].astype(str)+'_click_ratio_cnt'
        
        stat_click_1m_cnt=stat_click_1m.pivot(index='uid', columns='count_1m', values='click_count').reset_index().fillna(0)
        stat_click_1m_ratio_cnt=stat_click_1m.pivot(index='uid', columns='ratio_cnt_1m', values='ratio_cnt').reset_index().fillna(0)
        
        d_feature=pd.DataFrame({'uid':t_user.uid},index=t_user.index)
        d_feature=d_feature.merge(stat_click_1m_cnt,how="left", on="uid").fillna(0)
        d_feature=d_feature.merge(stat_click_1m_ratio_cnt,how="left", on="uid")
        if idx == 0:
            duser1 = duser1.merge(d_feature, how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(d_feature, how="left", on="uid")
    return duser1,duser2

def getLoanCntWithinOrder(duser1, duser2, window_size):
    valid_mask_loan, test_mask_loan = get_windows_mask(t_loan, "loan_time", window_size)
    valid_mask_buy, test_mask_buy = get_windows_mask(t_order, "buy_time", window_size)
    for idx, mask in enumerate([valid_mask_loan, test_mask_loan]):        
        tmp_loan = t_loan[mask]
        loanAmtWinds = tmp_loan.groupby(["uid","date"])["loan_amount"].sum().rename('loanAmtIn').reset_index()
        if idx==0:
            tmp_order = t_order[valid_mask_buy]
        elif idx==1:
            tmp_order = t_order[test_mask_buy]
        orderAmtWinds = tmp_order.groupby(["uid","buy_time"])["order_amt"].count().rename("orderCnt").reset_index()
        orderAmtWinds.columns = ["uid", "date", "orderCnt"]
        orderLoanAmtWinds = orderAmtWinds.merge(loanAmtWinds, how="left", on=["uid","date"])
        orderCntWinds= orderLoanAmtWinds[~orderLoanAmtWinds.loanAmtIn.isnull()].groupby("uid")['orderCnt'].sum().rename('loanOrderCnt' + str(window_size)).reset_index()
        orderTolCnt = orderAmtWinds.groupby("uid")['orderCnt'].sum().rename('orderTolCnt' + str(window_size)).reset_index()
        orderLoanAmtWinds = orderTolCnt.merge(orderCntWinds, how="left", on="uid")
        orderLoanAmtWinds['loanOrderCntRatioIn' + str(window_size)] = orderLoanAmtWinds['loanOrderCnt' + str(window_size)] /orderLoanAmtWinds['orderTolCnt' + str(window_size)]
        if idx == 0:
            duser1 = duser1.merge(orderLoanAmtWinds[["uid",'loanOrderCntRatioIn' + str(window_size)]], how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(orderLoanAmtWinds[["uid",'loanOrderCntRatioIn' + str(window_size)]], how="left", on="uid")
    return duser1, duser2

#if __name__ == '__main__':
work_path = "/Users/zhangkai/code/JDD_data"
#work_path = "/data/kai.zhang/JDD"
pd.set_option('display.max_columns', None)
os.getcwd() 
os.chdir(work_path) 
t_click = pd.read_csv("t_click.csv") 
t_user = pd.read_csv("t_user.csv")
t_loan = pd.read_csv("t_loan.csv") 
t_order = pd.read_csv("t_order.csv")
t_loan_sum = pd.read_csv("t_loan_sum.csv") 


t_click = parseDate(t_click, "click_time")
t_loan = parseDate(t_loan,"loan_time")
t_order = parseDate(t_order,"buy_time")
t_user = parseDate(t_user,"active_date")

t_loan["date"] = pd.to_datetime(t_loan["date"])
t_click["date"] = pd.to_datetime(t_click["date"])


t_click = t_click.groupby(["uid"]).apply(lambda x: x.sort_values(["click_time"], ascending=True)).reset_index(drop=True)
t_loan = t_loan.groupby(["uid"]).apply(lambda x: x.sort_values(["loan_time"], ascending=True)).reset_index(drop=True)
t_order = t_order.groupby(["uid"]).apply(lambda x: x.sort_values(["buy_time"], ascending=True)).reset_index(drop=True)
t_order["month"] = t_order["buy_time"].apply(lambda x: x.month)
t_loan["month"] = t_loan["loan_time"].apply(lambda x: x.month)
t_click["month"] = t_click["click_time"].apply(lambda x: x.month)

valid_end_date = "2016-11-01"
test_end_date = "2016-12-01"


##price解敏, t_loan和t_loan_sum是同一个脱敏函数
t_loan["loan_amount"] = t_loan.loan_amount.apply(lambda x: round(5**x -1))
t_order["price"] = t_order["price"].apply(lambda x: round(5**x -1))
t_order["discount"] = t_order["discount"].apply(lambda x: round(5**x -1))
t_user["limit"] = t_user["limit"].apply(lambda x: round(5**x -1))

##点击
#每个页面点击次数占比人均点击次数
tr_user, ts_user = click_percent(t_click, t_user, t_user, 30)

#用户点击的不同天数、页面id个数
tr_user, ts_user = click_days_pids(t_click, tr_user, ts_user, 30)

#页面点击平均点击间隔，最近一次点击距离现在的时间，下次点击的时间
tr_user, ts_user = getNearestClick(t_click, tr_user, ts_user, 30)


###购买
cate_mean_price = t_order[t_order.price != 0].groupby("cate_id")["price"].mean().rename("cate_mean_price").reset_index()
t_order = t_order.merge(cate_mean_price, on="cate_id", how="left")
t_order.loc[t_order.price<=0,"price"] = t_order.loc[t_order.price<=0,"cate_mean_price"]


#tmp = t_order[(t_order.qty > 10000)]
#t_order = t_order[t_order.price > 0].reset_index(drop=True)
t_order["order_amt"] = t_order["price"] * t_order["qty"] - t_order["discount"]
#t_order.loc[t_order["order_amt"]<0, "order_amt"] = 0
t_order.loc[t_order["order_amt"]<0, "order_amt"] = t_order.loc[t_order["order_amt"]<0, "order_amt"] + t_order.loc[t_order["order_amt"]<0, "discount"]

#t_order = t_order[t_order["order_amt"]>=0].reset_index(drop=True)

t_order["order_amt"] = t_order["order_amt"] + 1



#uid的购买次数（金额）占比人均购买次数（金额），uid平均每次购买金额，
tr_user, ts_user = uid_order_status(t_order, tr_user, ts_user, 30)


#购买平均间隔；最近一次购买金额，最近一次购买距离现在的时间，金额/时间； #最近一次购买距离用户激活的时间
tr_user, ts_user = getNearestOrder(t_order, tr_user, ts_user, 30)

#最大购买的金额，时间，金额/时间
tr_user, ts_user = getMaxPriceOrder(t_order, tr_user, ts_user, 30)



###借贷
t_loan['amt_per_plan']= t_loan['loan_amount']/t_loan['plannum'] 
t_loan['loan_interval'] = t_loan.groupby('uid')['loan_time'].diff().apply(lambda x:x.days+x.seconds/86400.0)
t_loan["pay_end_date"] = t_loan.apply(lambda x: x["loan_time"] + timedelta(days=x["plannum"] * 30), axis=1)  

##更新limit，10月单笔单款最大额度，即是11月份limit，11月份的贷款的最大单笔额度，12月份的limit，
tr_user, ts_user = updateLimit(tr_user, ts_user)


#贷款金额, 贷款期数, 每期贷款额, 贷款周期, 各次贷款离现在的时间相关的统计
tr_user, ts_user = gen_fixedtw_features_for_loan(t_loan, tr_user, ts_user, 30)

##激活时间距离现在的时间
tr_user["active_days"] = (pd.Timestamp(valid_end_date) - tr_user["active_date"]).apply(lambda x: x.days)
ts_user["active_days"] = (pd.Timestamp(test_end_date) - ts_user["active_date"]).apply(lambda x: x.days)

#最近一次借贷的金额, 最近一次借贷的金额/最近一次借贷距离现在的时间,最近一次贷款的分期数, 最近一次贷款的每期金额
tr_user, ts_user = getNearestLoan(t_loan, tr_user, ts_user)

#未来一个月要还多少钱
tr_user, ts_user = current2PayAmt(t_loan, tr_user, ts_user, 180)


##当前月尚未还贷情况，剩余贷款额度=limit - debt
tr_user, ts_user = currentDebtAmt(t_loan, tr_user, ts_user, 180)
tr_user, ts_user = currentDebtAmt(t_loan, tr_user, ts_user, 60)
#tr_user, ts_user = currentDebtAmt(t_loan, tr_user, ts_user, 30)

##join 11月总的借贷金额
t_loan_sum.drop("month", axis=1,inplace=True)
tr_user = tr_user.merge(t_loan_sum, on ="uid", how="left")
tr_user["loan_sum"] = tr_user["loan_sum"].fillna(0.0)


##每个月贷款金额/当月剩余额度
#当前剩余额度 = 初试额度 - 当月之前一共用了多少额度 + 已还额度
tr_user, ts_user = getLoanAmtRemainingLimt(t_loan, tr_user, ts_user, 1)
tr_user, ts_user = getLoanAmtRemainingLimt(t_loan, tr_user, ts_user, 2)  #当月剩余额度计算会有偏差

##上个月还了多少钱
tr_user, ts_user = lastMonthPayedAmt(t_loan, tr_user, ts_user)

##前三个月用户贷款的总金额、总次数，平均每次贷款金额，平均每月贷款金额,提升0.0011
tr_user, ts_user = getPast3MonthLoanFeatures(t_loan, tr_user, ts_user)

##最近一次,第一次贷款距离用户激活的时间, 最近一次,第一次贷款金额占比limit
tr_user, ts_user = loanTimeBetweenActivetime(t_loan, tr_user, ts_user)

##当前月还款金额/最近一次贷款距离现在的时间
tr_user["current_topay_amt_nearest_loantime"] = tr_user["current_topay_amt"]/(1+tr_user["nearest_loantime"])
ts_user["current_topay_amt_nearest_loantime"] = ts_user["current_topay_amt"]/(1+ts_user["nearest_loantime"])

##购买点击比
tr_user, ts_user = getOrderClickRatio(t_click, t_order, tr_user, ts_user)

#用户最近两次的贷款前后的间隔
tr_user, ts_user  = getNearest2LoanInterval(t_loan, tr_user, ts_user)

##用户购买的折扣率，重复计算了用户近三个月的购买金额
tr_user, ts_user = userDiscountRatio(t_order, tr_user, ts_user)

##离线效果微弱
#tr_user, ts_user  = getAvailableLoanAmtLimt(t_loan, tr_user, ts_user, 1)
#tr_user, ts_user  = getAvailableLoanAmtLimt(t_loan, tr_user, ts_user, 2)


##每人购买力，预测贷款金额
tr_user, ts_user = avgLoanAmt4orderAmt(t_loan, t_order, tr_user, ts_user)

##一阶二阶差分
tr_user, ts_user = getLoanShiftDiff(t_loan, tr_user, ts_user)
#tr_user, ts_user = getMonthLoanShiftDiff(t_loan, tr_user, ts_user)

tr_user, ts_user = getMonthOrderShiftDiff(t_order, tr_user, ts_user)

##loan，次数一阶二阶差分
tr_user, ts_user = getLoanCntShiftDiff(t_loan, tr_user, ts_user)


#tr_user["count_loanAmt_90_active_days"] = tr_user["count_loanAmt_90"]/tr_user["active_days"]
#ts_user["count_loanAmt_90_active_days"] = ts_user["count_loanAmt_90"]/ts_user["active_days"]


##性别-年龄-limit交叉
tr_user = getSexAgeLimt(t_user, tr_user)
ts_user = getSexAgeLimt(t_user, ts_user)

#tr_user, ts_user = currentMinDebtAmt(t_loan, tr_user, ts_user, 180)
#tr_user, ts_user = currentMaxDebtAmt(t_loan, tr_user, ts_user, 180)

tr_user, ts_user = getCatePivotAmtCnt(t_order, tr_user, ts_user)
tr_user, ts_user = uidOrderAmtCntWinds(t_order, tr_user, ts_user, 60)
tr_user, ts_user = uidOrderAmtCntWinds(t_order, tr_user, ts_user, 90)
tr_user, ts_user = getTotalPidStaytime(t_click, tr_user, ts_user)

##点击的页面展开
#tr_user, ts_user = gen_fixed_tw_features_for_click_PidParamUnFold(t_click, tr_user, ts_user, "click_time", 7)
#tr_user, ts_user = gen_fixed_tw_features_for_click_PidParamUnFold(t_click, tr_user, ts_user, "click_time", 14)
tr_user, ts_user = gen_fixed_tw_features_for_click_PidParamUnFold(t_click, tr_user, ts_user, "click_time", 30)
tr_user, ts_user = gen_fixed_tw_features_for_click_PidParamUnFold(t_click, tr_user, ts_user, "click_time", 60)
tr_user, ts_user = gen_fixed_tw_features_for_click_PidParamUnFold(t_click, tr_user, ts_user, "click_time", 90)

#tr_user,ts_user=gen_fixed_tw_features_for_loan_click_PidParamUnfold(t_loan,t_click,tr_user,ts_user,'loan_time',"click_time",30) 
#tr_user,ts_user=gen_fixed_tw_features_for_loan_click_PidParamUnfold(t_loan,t_click,tr_user,ts_user,'loan_time',"click_time",60) 
#tr_user,ts_user=gen_fixed_tw_features_for_loan_click_PidParamUnfold(t_loan,t_click,tr_user,ts_user,'loan_time',"click_time",90) 
#tr_user,ts_user=gen_fixed_tw_features_for_loan_click_PidParamUnfold(t_loan,t_click,tr_user,ts_user,'loan_time',"click_time",15) 

##没有贷款的日子里的点击情况
tr_user,ts_user=gen_fixed_tw_features_for_notloan_click_PidParamUnfold(t_loan,t_click,tr_user,ts_user,'loan_time',"click_time",30) 
tr_user,ts_user=gen_fixed_tw_features_for_notloan_click_PidParamUnfold(t_loan,t_click,tr_user,ts_user,'loan_time',"click_time",60) 
tr_user,ts_user=gen_fixed_tw_features_for_notloan_click_PidParamUnfold(t_loan,t_click,tr_user,ts_user,'loan_time',"click_time",90) 
tr_user,ts_user=gen_fixed_tw_features_for_notloan_click_PidParamUnfold(t_loan,t_click,tr_user,ts_user,'loan_time',"click_time",15) 


tr_user, ts_user = gen_fixed_tw_features_for_order_cateIdUnFold(t_order, tr_user, ts_user, "buy_time", 90)
tr_user, ts_user = gen_fixed_tw_features_for_order_cateIdUnFold(t_order, tr_user, ts_user, "buy_time", 60)
#tr_user, ts_user = gen_fixed_tw_features_for_order_cateIdUnFold(t_order, tr_user, ts_user, "buy_time", 30)
#tr_user, ts_user = gen_fixed_tw_features_for_order_cateIdUnFold(t_order, tr_user, ts_user, "buy_time", 15)


tr_user, ts_user = gen_fixed_tw_features_for_notOrder_click_PidParamUnfold(t_order, t_click, tr_user, ts_user,'buy_time',"click_time",30) 
tr_user, ts_user = gen_fixed_tw_features_for_notOrder_click_PidParamUnfold(t_order, t_click, tr_user, ts_user,'buy_time',"click_time",60) 
tr_user, ts_user = gen_fixed_tw_features_for_notOrder_click_PidParamUnfold(t_order, t_click, tr_user, ts_user,'buy_time',"click_time",90) 
tr_user, ts_user = gen_fixed_tw_features_for_notOrder_click_PidParamUnfold(t_order, t_click, tr_user, ts_user,'buy_time',"click_time",15) 


##购买点击序列的特征万分位rmse下降1个点
tr_user, ts_user = loanClickBehiviorSeries(tr_user, ts_user, 60)
#tr_user, ts_user = loanClickBehiviorSeries1(tr_user, ts_user, 60)

##购买次数、天数
tr_user, ts_user = orderCntDays(tr_user, ts_user, 30)
tr_user, ts_user = orderCntDays(tr_user, ts_user, 60)


###时间划分
valid_end_date = pd.Timestamp("2016-11-01")
valid_start_date= pd.Timestamp("2016-08-01")
valid_cut_point=[]
while valid_end_date>valid_start_date:
    valid_cut_point=valid_cut_point+[valid_end_date]
    valid_end_date=valid_end_date-timedelta(days=30)
valid_cut_point.sort()

test_end_date = pd.Timestamp("2016-12-01")
test_start_date= pd.Timestamp("2016-09-01")
test_cut_point=[]
while test_end_date>=test_start_date:
    test_cut_point=test_cut_point+[test_end_date]
    test_end_date=test_end_date-timedelta(days=30)   
test_cut_point.sort()    

l=len(valid_cut_point)
labels=['one_month_'+str(l-idx-1)for idx,j in enumerate(valid_cut_point) if idx+1<l]

tr_user, ts_user = gen_fixed_tw_features_for_click_1m(t_click,'click_time',tr_user,ts_user)


##备份
tr_user_bak = copy.deepcopy(tr_user)
ts_user_bak = copy.deepcopy(ts_user)


##还原
tr_user = copy.deepcopy(tr_user_bak)
ts_user = copy.deepcopy(ts_user_bak)



tr_user["next_loan"]= tr_user['mean_loanInterval_30'] + tr_user["nearest_loantime"]
ts_user["next_loan"]= tr_user['mean_loanInterval_30'] + ts_user["nearest_loantime"]

