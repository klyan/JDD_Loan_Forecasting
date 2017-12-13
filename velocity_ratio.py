# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:14:56 2017
@author: risk-public
"""
import copy
import pandas as pd
import numpy as np
import sys
import os
import pdb
import time
import gc
from datetime import datetime, timedelta,date
work_path = "D:/01_Study/07_JDDiscovery/code"
os.getcwd() 
os.chdir(work_path) 
'''
t_user = pd.read_csv("Loan_Forecasting_Qualification/t_user.csv") #90993
t_loan = pd.read_csv("Loan_Forecasting_Qualification/t_loan.csv") #202902
t_click = pd.read_csv("Loan_Forecasting_Qualification/t_click.csv") #10933016
t_loan = pd.read_csv("Loan_Forecasting_Qualification/t_loan.csv") #202902
t_order = pd.read_csv("Loan_Forecasting_Qualification/t_order.csv") #5400778
t_loan_sum = pd.read_csv("Loan_Forecasting_Qualification/t_loan_sum.csv") #19520
'''
def parseDate(df, col):
	df[col] = pd.to_datetime(df[col])
	if col == "click_time" or col == "loan_time" or col == "buy_time":
		df["date"] = df[col].apply(lambda x: x.date())
	return df


key='uid'
value='pid'
rec_tm='click_time'
velocity_df=t_click[[key,value,rec_tm]]
velocity_df['freq']=pd.Series([1]*t_click.shape[0],index=t_click.index)
time_window_list=[56]#目前只能一个,函数内不停的del，均是因为内存太小的缘故

velocity_df_bak = copy.deepcopy(velocity_df)

def velocity_ratio(key,value,rec_tm,time_window_list):
    ########local########################################################
    #基于key值统计的次数
    t_noDup=velocity_df[[key,rec_tm]].drop_duplicates()
    for tw in time_window_list:
        t_noDup['tw_start_indx_'+str(tw)]=pd.Series(['p_'+str(tw)+'_'+str(x) for x in list(t_noDup.index)],index=t_noDup.index)
    
    t_tw_start_indx=velocity_df[[key,rec_tm,'freq']].merge(t_noDup,how="left", on=[key,rec_tm])
                 
    t_v_df=t_tw_start_indx
    
    for tw in time_window_list:
        t_v_df_tw = pd.DataFrame({key:t_noDup[key], rec_tm:t_noDup[rec_tm]-timedelta(days=tw), 'freq':0})   #这个时间段的内的点击记录
        t_v_df_tw.index = pd.Series(['p_'+str(tw)+'_'+str(x) for x in list(t_v_df_tw.index)])
        t_v_df=pd.concat([t_v_df, t_v_df_tw])
    
    del t_noDup,t_v_df_tw
    gc.collect()
    
    t_v_cum=t_v_df.sort_values(by=[key,rec_tm]).groupby([key])['freq'].cumsum()
    t_v_cum=t_v_cum.reset_index()
    t_v_cum.rename(columns={'index':'index_0','freq':'cum_freq'}, inplace = True)
    t_vel_cum= t_tw_start_indx.merge(t_v_cum,how='left',left_index=True,right_on='index_0')
    t_vel_cum.rename(columns={'cum_freq':'cum_freq'+'_cur'}, inplace = True)
    t_vel_cum.drop('index_0',axis=1, inplace=True)          #当前点击事件，往前一共点击了多少次，没有时间限制
    
    del t_v_df
    gc.collect()
    
    for tw in time_window_list:
        t_vel_cum=t_vel_cum.merge(t_v_cum,how='left',left_on='tw_start_indx_'+str(tw),right_on='index_0')
        t_vel_cum.rename(columns={'cum_freq':'cum_freq_'+str(tw)}, inplace = True)
        t_vel_cum.drop('index_0',axis=1, inplace=True)
        
    cum_freq_cur=t_vel_cum.cum_freq_cur
    cum_freq_tw=t_vel_cum['cum_freq_'+str(tw)]

    del t_tw_start_indx,t_v_cum,t_vel_cum
    gc.collect()

    velocity_df['cnt_'+key+'_'+str(tw)+'_local'] =cum_freq_cur-cum_freq_tw
        
    del cum_freq_cur,cum_freq_tw
    gc.collect()

    #基于key value 统计的次数
    t_noDup=velocity_df[[key,value,rec_tm]].drop_duplicates()
    for tw in time_window_list:
        t_noDup['tw_start_indx_'+str(tw)]=pd.Series(['p_'+str(tw)+'_'+str(x) for x in list(t_noDup.index)],index=t_noDup.index)
    t_tw_start_indx=velocity_df[[key,value,rec_tm,'freq']].merge(t_noDup,how="left", on=[key,value,rec_tm])
             
    t_v_df=t_tw_start_indx
    
    for tw in time_window_list:
        t_v_df_tw=pd.DataFrame({key:t_noDup[key]
        ,value:t_noDup[value]
        ,rec_tm:t_noDup[rec_tm]-timedelta(days=tw)
        ,'freq':0
        })
        t_v_df_tw.index=pd.Series(['p_'+str(tw)+'_'+str(x) for x in list(t_v_df_tw.index)])
        t_v_df=pd.concat([t_v_df,t_v_df_tw])

    del t_noDup,t_v_df_tw
    gc.collect()
    
    t_v_cum=t_v_df.sort_values(by=[key,value,rec_tm]).groupby([key,value])['freq'].cumsum()
    t_v_cum=t_v_cum.reset_index()
    t_v_cum.rename(columns={'index':'index_0','freq':'cum_freq'}, inplace = True)
    t_vel_cum= t_tw_start_indx.merge(t_v_cum,how='left',left_index=True,right_on='index_0')
    t_vel_cum.rename(columns={'cum_freq':'cum_freq'+'_cur'}, inplace = True)
    t_vel_cum.drop('index_0',axis=1, inplace=True)
    
    del t_v_df
    gc.collect()
    
    for tw in time_window_list:
        t_vel_cum=t_vel_cum.merge(t_v_cum,how='left',left_on='tw_start_indx_'+str(tw),right_on='index_0')
        t_vel_cum.rename(columns={'cum_freq':'cum_freq_'+str(tw)}, inplace = True)
        t_vel_cum.drop('index_0',axis=1, inplace=True)
        
    cum_freq_cur=t_vel_cum.cum_freq_cur
    cum_freq_tw=t_vel_cum['cum_freq_'+str(tw)]

    del t_tw_start_indx,t_v_cum,t_vel_cum
    gc.collect()

    velocity_df['cnt_'+key+'_'+value+'_'+str(tw)+'_local'] =cum_freq_cur-cum_freq_tw
        
    del cum_freq_cur,cum_freq_tw
    gc.collect()
    
    ###########global#####################################################
    #基于全体统计的次数
    rec_tm_noDup=velocity_df[rec_tm].drop_duplicates()
    for tw in time_window_list:
        tw_start_indx=pd.Series(['p_'+str(tw)+'_'+str(x) for x in list(rec_tm_noDup.index)],index=rec_tm_noDup.index)
        t_noDup=pd.DataFrame({rec_tm:rec_tm_noDup,'tw_start_indx_'+str(tw):tw_start_indx},index=rec_tm_noDup.index)

    t_tw_start_indx=velocity_df[[rec_tm,'freq']].merge(t_noDup,how="left", on=[rec_tm])
            
    t_v_df=t_tw_start_indx
    
    for tw in time_window_list:
        t_v_df_tw=pd.DataFrame({ rec_tm:t_noDup[rec_tm]-timedelta(days=tw)
        ,'freq':0
        })
        t_v_df_tw.index=pd.Series(['p_'+str(tw)+'_'+str(x) for x in list(t_v_df_tw.index)])
        t_v_df=pd.concat([t_v_df,t_v_df_tw])
    
    del t_noDup,t_v_df_tw
    gc.collect()
    
    t_v_cum=t_v_df.sort_values(by=[rec_tm])['freq'].cumsum()
    t_v_cum=t_v_cum.reset_index()
    t_v_cum.rename(columns={'index':'index_0','freq':'cum_freq'}, inplace = True)
    t_vel_cum= t_tw_start_indx.merge(t_v_cum,how='left',left_index=True,right_on='index_0')
    t_vel_cum.rename(columns={'cum_freq':'cum_freq'+'_cur'}, inplace = True)
    t_vel_cum.drop('index_0',axis=1, inplace=True)
    
    del t_v_df
    gc.collect()
    
    for tw in time_window_list:
        t_vel_cum=t_vel_cum.merge(t_v_cum,how='left',left_on='tw_start_indx_'+str(tw),right_on='index_0')
        t_vel_cum.rename(columns={'cum_freq':'cum_freq_'+str(tw)}, inplace = True)
        t_vel_cum.drop('index_0',axis=1, inplace=True)
        
    cum_freq_cur=t_vel_cum.cum_freq_cur
    cum_freq_tw=t_vel_cum['cum_freq_'+str(tw)]

    del t_tw_start_indx,t_v_cum,t_vel_cum
    gc.collect()

    velocity_df['cnt_'+str(tw)+'_global'] =cum_freq_cur-cum_freq_tw
        
    del cum_freq_cur,cum_freq_tw
    gc.collect()
    
    #基于value统计的次数
    t_noDup=velocity_df[[value,rec_tm]].drop_duplicates()
    for tw in time_window_list:
        t_noDup['tw_start_indx_'+str(tw)]=pd.Series(['p_'+str(tw)+'_'+str(x) for x in list(t_noDup.index)],index=t_noDup.index)
    
    t_tw_start_indx=velocity_df[[value,rec_tm,'freq']].merge(t_noDup,how="left", on=[value,rec_tm])
                 
    t_v_df=t_tw_start_indx
    
    for tw in time_window_list:
        t_v_df_tw=pd.DataFrame({value:t_noDup[value]
        ,rec_tm:t_noDup[rec_tm]-timedelta(days=tw)
        ,'freq':0
        })
        t_v_df_tw.index=pd.Series(['p_'+str(tw)+'_'+str(x) for x in list(t_v_df_tw.index)])
        t_v_df=pd.concat([t_v_df,t_v_df_tw])
    
    del t_noDup,t_v_df_tw
    gc.collect()
    
    t_v_cum=t_v_df.sort_values(by=[value,rec_tm]).groupby([value])['freq'].cumsum()
    t_v_cum=t_v_cum.reset_index()
    t_v_cum.rename(columns={'index':'index_0','freq':'cum_freq'}, inplace = True)
    t_vel_cum= t_tw_start_indx.merge(t_v_cum,how='left',left_index=True,right_on='index_0')
    t_vel_cum.rename(columns={'cum_freq':'cum_freq'+'_cur'}, inplace = True)
    t_vel_cum.drop('index_0',axis=1, inplace=True)
    
    del t_v_df
    gc.collect()
    
    for tw in time_window_list:
        t_vel_cum=t_vel_cum.merge(t_v_cum,how='left',left_on='tw_start_indx_'+str(tw),right_on='index_0')
        t_vel_cum.rename(columns={'cum_freq':'cum_freq_'+str(tw)}, inplace = True)
        t_vel_cum.drop('index_0',axis=1, inplace=True)
        
    cum_freq_cur=t_vel_cum.cum_freq_cur
    cum_freq_tw=t_vel_cum['cum_freq_'+str(tw)]

    del t_tw_start_indx,t_v_cum,t_vel_cum
    gc.collect()

    velocity_df['cnt_'+value+'_'+str(tw)+'_local'] =cum_freq_cur-cum_freq_tw
        
    del cum_freq_cur,cum_freq_tw
    gc.collect()



######开始计算Velocity变量##########################################
t_click = pd.read_csv("Loan_Forecasting_Qualification/t_click.csv") #5400778
t_click = parseDate(t_click,'click_time')


key='uid'
value='pid'
rec_tm='click_time'
velocity_df=t_click[[key,value,rec_tm]]
velocity_df['freq']=pd.Series([1]*t_click.shape[0],index=t_click.index)
time_window_list=[56]#目前只能一个,函数内不停的del，均是因为内存太小的缘故

del t_click
gc.collect()

start = time.clock()
velocity_ratio(key,value,rec_tm,time_window_list)
end = time.clock()
print str(end-start)

