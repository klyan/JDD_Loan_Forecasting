##周期内购买金额与借贷金额差值, 离线变差一点
tr_user["gap_buy_loan_amt"] = tr_user["order_amt30"] - tr_user["sum_loanAmt_30"]
ts_user["gap_buy_loan_amt"] = ts_user["order_amt30"] - ts_user["sum_loanAmt_30"]


###估算剩余额度
tr_user, ts_user = getRemainingLoanAmt(t_loan, tr_user, ts_user)
tr_user["remainingAmt2"] = tr_user["limit"] - tr_user["unpayedLoanAmt"]
ts_user["remainingAmt2"] = ts_user["limit"] - ts_user["unpayedLoanAmt"]


def getRemainingLoanAmt(df_loan, duser1, duser2):
    valid_mask = df_loan.month < 11
    test_mask = df_loan.month < 12
    for idx, mask in enumerate([valid_mask, test_mask]):
        df_loan_tmp = df_loan[mask].reset_index(drop=True)
        payingLoan = df_loan_tmp.groupby("uid").apply(lambda x: x.sort_values("pay_end_date",ascending=True)).reset_index(drop=True)
        payingLoan["payedLoanAmt"]= payingLoan.groupby("uid")["loan_amount"].cumsum().rename("payedLoanAmt").reset_index(drop=True)
        historyLoanAmt = df_loan_tmp.groupby("uid")["loan_amount"].sum().rename("totalLoanAmt").reset_index()
        if idx ==0:
            historyLoanAmtLimit = historyLoanAmt.merge(duser1[["uid","limit"]], on="uid", how="left")
            historyLoanAmtLimit["payedAmt"] = historyLoanAmtLimit["totalLoanAmt"] - historyLoanAmtLimit["limit"]
            payedAmt = historyLoanAmtLimit.loc[historyLoanAmtLimit["payedAmt"] > 0,["uid","payedAmt"]].reset_index(drop=True)
            payingLoan = payingLoan.merge(payedAmt, on="uid", how="left")
            unpayedLoan = payingLoan[payingLoan["payedLoanAmt"] > payingLoan["payedAmt"]].reset_index(drop=True)
            unpayedLoanAmt = unpayedLoan.groupby("uid")["loan_amount"].sum().rename("unpayedLoanAmt").reset_index()
            duser1 = duser1.merge(unpayedLoanAmt, on="uid", how="left")
        elif idx == 1:
            historyLoanAmtLimit = historyLoanAmt.merge(duser2[["uid","limit"]], on="uid", how="left")
            historyLoanAmtLimit["payedAmt"] = historyLoanAmtLimit["totalLoanAmt"] - historyLoanAmtLimit["limit"]
            payedAmt = historyLoanAmtLimit.loc[historyLoanAmtLimit["payedAmt"] > 0,["uid","payedAmt"]].reset_index(drop=True)
            payingLoan = payingLoan.merge(payedAmt, on="uid", how="left")
            unpayedLoan = payingLoan[payingLoan["payedLoanAmt"] > payingLoan["payedAmt"]].reset_index(drop=True)
            unpayedLoanAmt = unpayedLoan.groupby("uid")["loan_amount"].sum().rename("unpayedLoanAmt").reset_index()
            duser2 = duser2.merge(unpayedLoanAmt, on="uid", how="left")
    return duser1, duser2

tr_user = encodeByuserProfile(tr_user, 10)
ts_user = encodeByuserProfile(ts_user, 11)

##人群编码
def encodeByuserProfile(duser, monthlimit):
    duser["encode_key"]= duser["sex_age_limit"].astype(str) + duser["limit_increase"].apply(lambda x: round(x,1)).astype(str)
    monthLoanAmt = t_loan.groupby(["uid","month"],as_index=False)["loan_amount"].agg(sum)
    encodeMean= monthLoanAmt[monthLoanAmt.month <= monthlimit].groupby(["uid"],as_index=False)["loan_amount"].mean()
    encodeMean.columns = ["uid", "monthLoanMeanAmt"]
    encodeMean = encodeMean.merge(duser[["uid","encode_key"]], on="uid", how="left")
    encodeMean = encodeMean.groupby("encode_key")["monthLoanMeanAmt"].mean().rename("encode_mean").reset_index()
    duser = duser.merge(encodeMean, on="encode_key", how="left")
    duser.drop("encode_key", axis=1, inplace=True)
    return duser

##前三个月每个月贷款金额的置信度
def getConfidenceInterval(duser1, duser2, degree):
    # mean +/- z * std/sqrt(n)
    if degree == 0.8:
        z = 1.282
    elif degree == 0.85:
        z = 1.440
    elif degree == 0.90:
        z = 1.645
    elif degree ==0.95:
        z = 1.960
    valid_mask = t_loan.month<11
    test_mask = t_loan.month<12
    for idx, mask in enumerate([valid_mask, test_mask]):
        month_loanAmt = t_loan[mask].groupby(["uid","month"])['loan_amount'].sum().rename("monthLoanAmt").reset_index()
        month_loanAmt_status = month_loanAmt.groupby(["uid"])["monthLoanAmt"].agg(["count","mean","std"]).reset_index().fillna(0.0)
        month_loanAmt_status["upperConfidence"] = month_loanAmt_status["mean"] + z * month_loanAmt_status["std"] / month_loanAmt_status["count"].apply(lambda x: sqrt(x))
        month_loanAmt_status["lowerConfidence"] = month_loanAmt_status["mean"] - z * month_loanAmt_status["std"] / month_loanAmt_status["count"].apply(lambda x: sqrt(x))
        month_loanAmt_status.columns = ["uid", "loanMonthCount","loanMonthAmtMean","loanMonthAmtStd","upperConfidence","lowerConfidence"]
        if idx ==0:
            duser1 = duser1.merge(month_loanAmt_status, on="uid", how="left")
        elif idx ==1:
            duser2 = duser2.merge(month_loanAmt_status, on="uid", how="left")
    return duser1, duser2

def getOrderAmtWinds(df, duser, tuser, window_size):
    valid_mask, test_mask = get_windows_mask(df, "buy_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        uid_order = df[mask].groupby(["uid"])["order_amt"].sum().rename("order_amt_" + str(window_size)).reset_index()
        if idx == 0:
            duser = duser.merge(uid_order, how="left", on = 'uid')
        elif idx == 1:
            tuser = tuser.merge(uid_order, how="left", on = 'uid')
    return duser, tuser


##计算每个页面的停留时间
tr_user, ts_user = getStaytimePid(t_click, tr_user, ts_user)
def getStaytimePid(df_click, duser1, duser2):
    valid_mask = df_click.month.isin([10,8,9])
    test_mask = df_click.month.isin([11,10,9])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df_click[mask].groupby(["uid","last_click_pid"])["click_interval"].sum().rename("pidStaytime").reset_index()
        tmp1 = tmp.pivot(index='uid', columns='last_click_pid', values='pidStaytime').reset_index().fillna(0)
        tmp1.columns =  ['uid']+ [ 'pid_' + str(i) +'_staytime' for i in list(tmp1.columns)[1:]]
        tmp1.drop("pid_0.0_staytime", inplace=True, axis=1)
        if idx == 0:
            duser1 = duser1.merge(tmp, on ="uid", how="left")
        elif idx == 1:   
            duser2 = duser2.merge(tmp, on ="uid", how="left")
    return duser1, duser2

tr_user, ts_user = getTotalPidStaytime(t_click, tr_user, ts_user)

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

tr_user, ts_user= uidOrderAmtCntWinds(t_order, tr_user, ts_user, 90)

tr_user, ts_user= uidOrderAmtCntWinds(t_order, tr_user, ts_user, 60)

tr_user, ts_user= uidOrderAmtCntWinds(t_order, tr_user, ts_user, 30)

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


tr_user, ts_user= uidOrderLoanAmtWinds(tr_user, ts_user, 30)

tr_user, ts_user= uidOrderLoanAmtWinds( tr_user, ts_user, 60)

tr_user, ts_user= uidOrderLoanAmtWinds( tr_user, ts_user, 90)


def uidOrderLoanAmtWinds( duser, tuser, window_size):
    valid_ordermask, test_ordermask = get_windows_mask(t_order, "buy_time", window_size)
    valid_loanmask, test_loanmask = get_windows_mask(t_loan, "loan_time", window_size)
    for idx, mask in enumerate([valid_ordermask, test_ordermask]):
        tmp = t_order[mask]
        order_status = tmp.groupby(["uid"])["order_amt"].sum().rename("orderAmt" + str(window_size)).reset_index()
        if idx == 0:
            loanstatus = t_loan[valid_loanmask].groupby(["uid"])["loan_amount"].sum().rename("loanAmt" + str(window_size)).reset_index()
            order_loan = order_status.merge(loanstatus,on="uid",how="left")
            order_loan["loanOrderGapAmt" + str(window_size)] = order_loan["orderAmt" + str(window_size)] - order_loan["loanAmt" + str(window_size)]
            duser = duser.merge(order_loan[["uid","loanOrderGapAmt" + str(window_size)]], how="left", on = 'uid')
        elif idx == 1:
            loanstatus = t_loan[test_loanmask].groupby(["uid"])["loan_amount"].sum().rename("loanAmt" + str(window_size)).reset_index()
            order_loan = order_status.merge(loanstatus,on="uid",how="left")
            order_loan["loanOrderGapAmt" + str(window_size)] = order_loan["orderAmt" + str(window_size)] - order_loan["loanAmt" + str(window_size)]
            tuser = tuser.merge(order_loan[["uid","loanOrderGapAmt" + str(window_size)]], how="left", on = 'uid')
    return duser, tuser



def getLoanIntervalShiftDiff(df_loan, duser1, duser2):
    valid_mask = df_loan.month.isin([8,  9, 10])
    test_mask = df_loan.month.isin([8, 9, 10, 11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df_loan[mask].reset_index(drop=True)
        tmp["loanIntervalDiff"] = tmp.groupby("uid")["loan_interval"].apply(lambda x: x - x.shift(1))
        tmp["loanIntervalDiff2"] = tmp.groupby("uid")["loanIntervalDiff"].apply(lambda x: x - x.shift(1))
        maxtime_idx = tmp.groupby(['uid'])['loan_time'].transform(max) == tmp['loan_time']  #用户最近一天贷款的情况
        tmp = tmp[maxtime_idx]
        if idx == 0:
            duser1 = duser1.merge(tmp[["uid", "loanIntervalDiff","loanIntervalDiff2"]], on ="uid", how="left")
        elif idx == 1:
            duser2 = duser2.merge(tmp[["uid", "loanIntervalDiff","loanIntervalDiff2"]], on ="uid", how="left")
    return duser1, duser2    

def getMonthLoanShiftDiff_WRONG(df_loan, duser1, duser2):
    uidMonthLoan = df_loan.groupby(["uid","month"])["loan_amount"].sum().rename("month_loan_amt").reset_index()
    uidMonthLoan = uidMonthLoan.pivot(index='uid', columns='month', values='month_loan_amt').fillna(0)
    uidMonthLoan = uidMonthLoan.stack().reset_index()
    uidMonthLoan.columns =["uid","month","month_loan_amt"]
    uidMonthLoan = uidMonthLoan.groupby(["uid"]).apply(lambda x: x.sort_values(["month"], ascending=True)).reset_index(drop=True)
    uidMonthLoan["monthLoanAmtDiff"] = uidMonthLoan.groupby("uid")["month_loan_amt"].apply(lambda x: x - x.shift(1))
    uidMonthLoan["monthLoanAmtDiff2"] = uidMonthLoan.groupby("uid")["monthLoanAmtDiff"].apply(lambda x: x - x.shift(1))
    duser1 = duser1.merge(uidMonthLoan[uidMonthLoan.month == 10][["uid", "monthLoanAmtDiff","monthLoanAmtDiff2"]], on ="uid", how="left")
    duser2 = duser2.merge(uidMonthLoan[uidMonthLoan.month == 11][["uid", "monthLoanAmtDiff","monthLoanAmtDiff2"]], on ="uid", how="left")
    return duser1, duser2

##最近一次借款时间-与最近一次购买时间差
tr_user, ts_user = getNearestLoanOrderInterval(t_loan, t_order, tr_user, ts_user)
def getNearestLoanOrderInterval(df_loan, df_order, duser1, duser2):
    window_size = 180
    loan_valid_mask, loan_test_mask = get_windows_mask(df_loan, "loan_time", window_size)
    order_valid_mask, order_test_mask = get_windows_mask(df_order, "buy_time", window_size)
    for i in [0,1]:
        if i == 0:
            loan_tmp = df_loan[loan_valid_mask][["uid","loan_time","loan_amount"]]
            order_tmp = df_order[order_valid_mask][["uid","buy_time","order_amt"]]        
        elif i == 1:
            loan_tmp = df_loan[loan_test_mask][["uid","loan_time","loan_amount"]]
            order_tmp = df_order[order_test_mask][["uid","buy_time","order_amt"]]
        uid_nearest_loan = loan_tmp[loan_tmp.groupby(['uid'])['loan_time'].transform(max) == loan_tmp['loan_time']]  #用户最近一次贷款的时间
        uid_nearest_order = order_tmp[order_tmp.groupby(['uid'])['buy_time'].transform(max) == order_tmp['buy_time']]
        uid_loan_order = uid_nearest_loan.merge(uid_nearest_order, on = 'uid', how='left')
        uid_loan_order["nearest_order_loan_interval"] = (uid_loan_order["loan_time"] - uid_loan_order["buy_time"]).apply(lambda x: x.days)
        uid_loan_order["nearest_order_loan_price"] = (uid_loan_order["loan_amount"] - uid_loan_order["order_amt"])
        if i == 0:
            duser1 = duser1.merge(uid_loan_order[["uid","nearest_order_loan_interval","nearest_order_loan_price"]], on="uid", how="left")
        elif i == 1:   
            duser2 = duser2.merge(uid_loan_order[["uid","nearest_order_loan_interval","nearest_order_loan_price"]], on="uid", how="left")
    return duser1, duser2

##用户当前所有贷款的未还款之前贷了多少钱
tr_user, ts_user = loanAgainBeforePayed(tr_user, ts_user, 15)
tr_user, ts_user = loanAgainBeforePayed(tr_user, ts_user, 30)

def loanAgainBeforePayed(duser1, duser2, window_size):
    valid_mask_loan, test_mask_loan = get_windows_mask(t_loan, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask_loan, test_mask_loan]):
        tmp = t_loan[mask].groupby("uid")["loan_amount"].sum().rename("loanAmount").reset_index()
        if idx == 0:
            valid_before = pd.Timestamp(valid_end_date) - timedelta(days=window_size)
            still_mask = (t_loan["loan_time"] < valid_before) & (t_loan["pay_end_date"] >= pd.Timestamp(valid_end_date))
            stillLoan = t_loan[still_mask].reset_index(drop=True)
            stillLoan["topayed_num"] = (stillLoan["pay_end_date"] - pd.Timestamp(valid_end_date)).apply(lambda x: x.days/30.0)
        elif idx ==1:
            test_before = pd.Timestamp(test_end_date) - timedelta(days=window_size)
            still_mask = (t_loan["loan_time"] < valid_before) & (t_loan["pay_end_date"] >= pd.Timestamp(test_end_date))
            stillLoan = t_loan[still_mask].reset_index(drop=True)
            stillLoan["topayed_num"] = (stillLoan["pay_end_date"] - pd.Timestamp(test_end_date)).apply(lambda x: x.days/30.0)
        stillLoan["debtAmt"] = stillLoan["amt_per_plan"] * stillLoan["topayed_num"]
        stillLoan = stillLoan.groupby("uid")["debtAmt"].sum().rename("stillDebtAmt").reset_index()
        tmp = tmp.merge(stillLoan, on="uid", how="left")
        tmp["stillDebtAmt_" + str(window_size)] = tmp["loanAmount"] + tmp["stillDebtAmt"]
        if idx == 0:
            duser1 = duser1.merge(tmp[["uid","stillDebtAmt_" + str(window_size)]], on="uid", how="left")
        elif idx == 1:   
            duser2 = duser2.merge(tmp[["uid","stillDebtAmt_" + str(window_size)]], on="uid", how="left")
    return duser1, duser2



tr_user, ts_user = loanOrderBehiviorSeries(tr_user, ts_user, 30)

def loanOrderBehiviorSeries(duser1,duser2,window_size):
    valid_mask_loan, test_mask_loan = get_windows_mask(t_loan, "loan_time", window_size)
    valid_mask_order, test_mask_order = get_windows_mask(t_order, "buy_time", window_size)
    for idx, mask in enumerate([valid_mask_loan, test_mask_loan]):
        tmp_loan = t_loan.loc[mask,['uid','date']].rename(columns={'date':'behavior_time'})
        tmp_loan['behavior'] = 0
        if idx == 0:
            mask_order = valid_mask_order
        elif idx == 1:
            mask_order = test_mask_order 
        tmp_order = t_order.loc[mask_order,['uid','buy_time']].rename(columns={'buy_time':'behavior_time'})
        tmp_order['behavior']=1
        tmp=pd.concat([tmp_loan.drop_duplicates(),tmp_order.drop_duplicates()]).groupby(["uid"]).apply(lambda x: x.sort_values(["behavior_time"], ascending=True)).reset_index(drop=True)
        tmp1 = tmp.groupby("uid")["behavior"].apply(lambda x:list(x)).rename("orderLoanBehavior").reset_index()
        tmp1["orderLoanBehavior"] = tmp1["orderLoanBehavior"].apply(lambda x: ''.join(map(str, x)))
        tmp1["orderLoanBehavior"] = tmp1["orderLoanBehavior"].astype('category')
        tmp1['orderLoanBehavior'].cat.categories=range(tmp1["orderLoanBehavior"].nunique())  #4563
        tmp1["orderLoanBehavior"] = tmp1["orderLoanBehavior"].astype(int)
        if idx == 0:
            duser1 = duser1.merge(tmp1[["uid","orderLoanBehavior"]], on="uid", how="left")
        elif idx == 1:
            duser2 = duser2.merge(tmp1[["uid","orderLoanBehavior"]], on="uid", how="left")
    return duser1, duser2


tr_user, ts_user = getOrderLoanAmtdiff(tr_user, ts_user, 60)

##购买到借贷的平均时间间隔、价格差
def getOrderLoanAmtdiff(duser1, duser2, window_size):
    tmp_loan = t_loan[["uid","loan_time","loan_amount"]] 
    tmp_order = t_order[["uid","buy_time","order_amt"]] 
    tmp_loan["type"] = "loan"
    tmp_order["type"] = "order"
    tmp_loan.columns = ["uid","action_time","amount","type"]
    tmp_order.columns = ["uid","action_time","amount","type"]
    loan_order = pd.concat([tmp_loan,tmp_order])
    loan_order = loan_order.sort_values(["action_time"], ascending=True).reset_index(drop=True)
    valid_mask, test_mask = get_windows_mask(loan_order,"action_time",window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = loan_order[mask].reset_index(drop=True)
        tmp['last_action_time']= tmp.groupby(['uid'])[['action_time']].shift(1)
        tmp['last_type']= tmp.groupby(['uid'])[['type']].shift(1)
        tmp["order_click_interval"] = (tmp["action_time"] - tmp['last_action_time']).apply(lambda x: x.days)
        uid_order_click= tmp[tmp["type"] != tmp["last_type"]].groupby("uid")["order_click_interval"].agg(["mean","max","min"]).reset_index()
        uid_order_click.columns = ["uid","mean_order_click_interval","max_order_click_interval","min_order_click_interval"]
        if idx == 0:
            duser1 = duser1.merge(uid_order_click, on="uid", how="left")
        elif idx == 1:
            duser2 = duser2.merge(uid_order_click, on="uid", how="left")
    return duser1, duser2


def getOrderLoanAmtdiff(duser1, duser2, window_size):
    tmp_loan = t_loan[["uid","loan_time","loan_amount"]] 
    tmp_order = t_order[["uid","buy_time","order_amt"]] 
    tmp_loan["type"] = "loan"
    tmp_order["type"] = "order"
    tmp_loan.columns = ["uid","action_time","amount","type"]
    tmp_order.columns = ["uid","action_time","amount","type"]
    loan_order = pd.concat([tmp_loan,tmp_order])
    loan_order = loan_order.sort_values(["action_time"], ascending=True).reset_index(drop=True)
    valid_mask, test_mask = get_windows_mask(loan_order,"action_time",window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = loan_order[mask].reset_index(drop=True)
        tmp['last_amount']= tmp.groupby(['uid'])[['amount']].shift(1)
        tmp['last_type']= tmp.groupby(['uid'])[['type']].shift(1)
        tmp["order_click_gap"] = (tmp["amount"] - tmp['last_amount'])
        uid_order_click= tmp[tmp["type"] != tmp["last_type"]].groupby("uid")["order_click_gap"].agg(["mean","max","min"]).reset_index()
        uid_order_click.columns = ["uid","mean_order_click_interval","max_order_click_interval","min_order_click_interval"]
        if idx == 0:
            duser1 = duser1.merge(uid_order_click, on="uid", how="left")
        elif idx == 1:
            duser2 = duser2.merge(uid_order_click, on="uid", how="left")
    return duser1, duser2

##用户一个月每个月每个类目消费金额、次数, 时间周期分了三次
tr_user, ts_user = getCatePivotAmtCnt(t_order, tr_user, ts_user)

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

##用户前三个月每个月消费，贷款、金额、次数
def getPivotAmtCnt(df, column ,duser1, duser2):
    valid_mask = df.month.isin([10,9,8])
    test_mask = df.month.isin([11,10,9])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask]
        uid_months = tmp.groupby(["uid","month"])[column].agg(["count","sum"]).reset_index()
        uid_months.rename({'count': column + '_cnt', 'sum': column + '_sum' }, axis='columns',inplace=True)
        if idx == 0:
            uid_months["month"]  = "month" + uid_months['month'].astype(str)
        elif idx == 1:
            uid_months["month"]  = "month" + (uid_months['month']-1).astype(str)
        uid_months = uid_months.pivot(index='uid', columns='month').reset_index().fillna(0)
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

##用户点击多少商品, 效果不好，下降0.001
tr_user, ts_user = getClick(t_click, tr_user, ts_user)

def getClick(df_click, duser1, duser2):
    df_click["pid_param"] = df_click["pid"].astype(str) + "_" + df_click["param"].astype(str)
    valid_mask = df_click.month.isin([8,9,10])
    test_mask = df_click.month.isin([9,10,11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df_click[mask]
        product = tmp.groupby("uid")["pid_param"].nunique().rename("click_pdt_cnt").reset_index()
        if idx == 0:
            duser1 = duser1.merge(product, on="uid", how="left")
        elif idx == 1:
            duser2 = duser2.merge(product, on="uid", how="left")
    return duser1, duser2


##按天汇总loan order金额， 1，7，15，30
tr_user, ts_user = getLoanAmtPerWindows(t_loan, tr_user, ts_user, 7)

tr_user, ts_user = getLoanAmtPerWindows(t_loan, tr_user, ts_user, 15)

def getLoanAmtPerWindows(df_loan, duser1, duser2, window_size):
    valid_mask = df_loan.month < 11
    test_mask = df_loan.month < 12
    df_loan["dayofyear"] = df_loan["loan_time"].apply(lambda x: x.dayofyear)
    for idx, mask in enumerate([valid_mask, test_mask]):
        df_loan_tmp = df_loan[mask].reset_index(drop=True)
        df_loan_tmp["dayofwindows"] = df_loan_tmp["dayofyear"].apply(lambda x: round(x/window_size))
        uid_tmp = df_loan_tmp.groupby(["uid","dayofwindows"])["loan_amount"].sum().rename("loanAmtIn"+str(window_size)).reset_index().groupby("uid")["loanAmtIn"+str(window_size)].mean().reset_index()
        if idx == 0:
            duser1 = duser1.merge(uid_tmp, on="uid", how="left")
            duser1["loanAmtIn"+str(window_size)] = duser1["loanAmtIn"+str(window_size)].fillna(0)
        elif idx == 1:
            duser2 = duser2.merge(uid_tmp, on="uid", how="left")
            duser2["loanAmtIn"+str(window_size)] = duser2["loanAmtIn"+str(window_size)].fillna(0)
    return duser1, duser2


##最近五次的贷款金额、时间差、金额/时间差、期数
tr_user, ts_user = getNthNearestLoan(tr_user, ts_user, 2)
tr_user, ts_user = getNthNearestLoan(tr_user, ts_user, 3)
tr_user, ts_user = getNthNearestLoan(tr_user, ts_user, 4)
tr_user, ts_user = getNthNearestLoan(tr_user, ts_user, 5)

def getNthNearestLoan(duser1, duser2, pos):
    loan_validmask = t_loan.month < 11
    loan_testmask = t_loan.month < 12
    for idx, mask in enumerate([loan_validmask, loan_testmask]):
        t_loan_tmp = t_loan[mask].reset_index(drop=True)
        t_loan_tmp["loan_rank"] = t_loan_tmp.groupby(["uid"])["loan_time"].rank(ascending=False)
        t_loan_tmp = t_loan_tmp[t_loan_tmp["loan_rank"] == pos][["uid","loan_time","loan_amount","plannum"]]
        t_loan_tmp.columns = ["uid","loan_time", "pos_" + str(pos) + "_loan_amount", "plannum"]        
        if idx ==0:
            t_loan_tmp["pos_"+str(pos)+"_loan_interval"] = (pd.Timestamp(valid_end_date)-t_loan_tmp["loan_time"]).apply(lambda x:x.total_seconds())
            t_loan_tmp["loan_amount_interval_pos_" + str(pos)]= t_loan_tmp["pos_" + str(pos) + "_loan_amount"] / t_loan_tmp["pos_"+str(pos)+"_loan_interval"]
            t_loan_tmp = t_loan_tmp[["uid", "pos_" + str(pos) + "_loan_amount", "pos_"+str(pos)+"_loan_interval", "loan_amount_interval_pos_" + str(pos)]]
            duser1 = duser1.merge(t_loan_tmp, on="uid", how="left")
        elif idx == 1:
            t_loan_tmp["pos_"+str(pos)+"_loan_interval"] = (pd.Timestamp(test_end_date)-t_loan_tmp["loan_time"]).apply(lambda x:x.total_seconds())
            t_loan_tmp["loan_amount_interval_pos_" + str(pos)]= t_loan_tmp["pos_" + str(pos) + "_loan_amount"] / t_loan_tmp["pos_"+str(pos)+"_loan_interval"]
            t_loan_tmp = t_loan_tmp[["uid", "pos_" + str(pos) + "_loan_amount", "pos_"+str(pos)+"_loan_interval", "loan_amount_interval_pos_" + str(pos)]]
            duser2 = duser2.merge(t_loan_tmp, on="uid", how="left")
    return duser1, duser2


##最近N的点击记录

tr_user, ts_user = getNthNearestClick(tr_user, ts_user, 1)
tr_user, ts_user = getNthNearestClick(tr_user, ts_user, 2)
tr_user, ts_user = getNthNearestClick(tr_user, ts_user, 3)
tr_user, ts_user = getNthNearestClick(tr_user, ts_user, 4)
tr_user, ts_user = getNthNearestClick(tr_user, ts_user, 5)


def getNthNearestClick(duser1, duser2, pos):
    click_validmask = t_click.month < 11
    click_testmask = t_click.month < 12
    for idx, mask in enumerate([click_validmask, click_testmask]):
        t_click_tmp = t_click[mask].reset_index(drop=True)
        t_click_tmp["pid_param"]= t_click_tmp["pid"].astype(str) + "_" + t_click_tmp["param"].astype(str)
        t_click_tmp["loan_rank"] = t_click_tmp.groupby(["uid"])["click_time"].rank(ascending=False)
        t_click_tmp = t_click_tmp[t_click_tmp["loan_rank"] == pos][["uid","pid_param"]]
        t_click_tmp.columns = ["uid", "pid_param_pos" + str(pos)]
        if idx ==0:
            duser1 = duser1.merge(t_click_tmp, on="uid", how="left")
            duser1["pid_param_pos" + str(pos)] = duser1["pid_param_pos" + str(pos)].astype('category')
            duser1["pid_param_pos" + str(pos)].cat.categories= np.arange(1,duser1["pid_param_pos" + str(pos)].nunique()+1)
            duser1["pid_param_pos" + str(pos)] = duser1["pid_param_pos" + str(pos)].astype(int)
        elif idx == 1:
            duser2 = duser2.merge(t_click_tmp, on="uid", how="left")
            duser2["pid_param_pos" + str(pos)] = duser2["pid_param_pos" + str(pos)].astype('category')
            duser2["pid_param_pos" + str(pos)].cat.categories= np.arange(1,duser2["pid_param_pos" + str(pos)].nunique()+1)
            duser2["pid_param_pos" + str(pos)] = duser2["pid_param_pos" + str(pos)].astype(int)
    return duser1, duser2


##每人购买力，预测贷款金额
def getRealPurchasePower(df_loan, df_order,  duser1, duser2):
    month_orderAmt = df_order.groupby(["uid","month"])["order_amt"].sum().rename("uid_month_orderAmt").reset_index()
    month_loanAmt = df_loan.groupby(["uid","month"])["loan_amount"].sum().rename("uid_month_loanAmt").reset_index()
    month_order_loan = month_orderAmt.merge(month_loanAmt, on = ["uid","month"], how="left").fillna(0.0)
    month_order_loan["realPurchasePower"] = month_order_loan["uid_month_orderAmt"] - month_order_loan["uid_month_loanAmt"]  
    realPurchasePower = month_order_loan[month_order_loan.month < 11]
    tmp = realPurchasePower.groupby("uid")["realPurchasePower"].agg(["max","min","mean","median"]).reset_index()
    tmp.columns = ["uid","max_realPurchasePower", "min_realPurchasePower", "mean_realPurchasePower", "median_realPurchasePower"]
    duser1 = duser1.merge(tmp, on = 'uid', how="left")
    tmp = month_order_loan.groupby("uid")["realPurchasePower"].agg(["max","min","mean","median"]).reset_index()
    tmp.columns = ["uid","max_realPurchasePower", "min_realPurchasePower", "mean_realPurchasePower", "median_realPurchasePower"]
    duser2 = duser2.merge(tmp, on = 'uid', how="left")
    return duser1, duser2

##每人购买力，预测贷款金额
def avgLoanAmt4orderAmt(df_loan, df_order,  duser1, duser2):
    month_orderAmt = df_order.groupby(["uid","month"])["order_amt"].sum().rename("uid_month_orderAmt").reset_index()
    month_loanAmt = df_loan.groupby(["uid","month"])["loan_amount"].sum().rename("uid_month_loanAmt").reset_index()
    month_order_loan = month_orderAmt.merge(month_loanAmt, on = ["uid","month"], how="left").fillna(0.0)
    valid_mask = month_order_loan.month < 11
    test_mask = month_order_loan.month < 12
    for idx, mask in enumerate([valid_mask, test_mask]):
        monthOrderLoanTmp = month_order_loan[mask].reset_index(drop=True)
        monthOrderLoanTmp["orderLoanAmtGap"] = monthOrderLoanTmp["uid_month_loanAmt"] + monthOrderLoanTmp["uid_month_orderAmt"]
        meanOrderLoanAmtGap = monthOrderLoanTmp.groupby("uid")["orderLoanAmtGap"].mean().rename("meanOrderLoanAmtGap").reset_index()
        if idx == 0:
            duser1 = duser1.merge(meanOrderLoanAmtGap, on = 'uid', how="left")
        elif idx == 1:
            duser2 = duser2.merge(meanOrderLoanAmtGap, on = 'uid', how="left")
    return duser1, duser2

##每个人最近一次贷款之后购买金额，次数, rmse微弱上升
def getOrderStatusAfterNearestLoan(df_loan, df_order, duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(df_loan, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        df_loan_tmp = df_loan[mask].reset_index(drop=True)
        maxtime_idx = df_loan_tmp.groupby(['uid'])['loan_time'].transform(max) == df_loan_tmp['loan_time']  #用户最近一次贷款的情况
        uid_nearest_loan = df_loan_tmp[maxtime_idx].reset_index(drop=True)
        order_loan = df_order.merge(uid_nearest_loan[["uid","loan_time"]], on="uid", how="left")
        if idx == 0:
            valid_idx = (order_loan["buy_time"] > order_loan["loan_time"]) & (order_loan["buy_time"] < pd.Timestamp("2016-11-01"))
            afterLoanOrderStatus = order_loan[valid_idx].groupby("uid")["order_amt"].agg(["count","sum"]).reset_index()
            afterLoanOrderStatus.columns = ["uid","afterNearestLoanOrderCnt", "afterNearestLoanOrderAmt"]
            duser1 = duser1.merge(afterLoanOrderStatus, on="uid", how="left")
            duser1[["afterNearestLoanOrderCnt", "afterNearestLoanOrderAmt"]] = duser1[["afterNearestLoanOrderCnt", "afterNearestLoanOrderAmt"]].fillna(0)
        elif idx ==1:
            valid_idx = (order_loan["buy_time"] > order_loan["loan_time"]) & (order_loan["buy_time"] < pd.Timestamp("2016-12-01"))
            afterLoanOrderStatus = order_loan[valid_idx].groupby("uid")["order_amt"].agg(["count","sum"]).reset_index()
            afterLoanOrderStatus.columns = ["uid","afterNearestLoanOrderCnt", "afterNearestLoanOrderAmt"]
            duser2 = duser2.merge(afterLoanOrderStatus, on="uid", how="left")
            duser2[["afterNearestLoanOrderCnt", "afterNearestLoanOrderAmt"]] = duser2[["afterNearestLoanOrderCnt", "afterNearestLoanOrderAmt"]].fillna(0)
    return duser1, duser2


##每个人最近一次贷款之后点击的次数, 
def getClickStatusAfterNearestLoan(df_loan, df_click, duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(df_loan, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        df_loan_tmp = df_loan[mask].reset_index(drop=True)
        maxtime_idx = df_loan_tmp.groupby(['uid'])['loan_time'].transform(max) == df_loan_tmp['loan_time']  #用户最近一次贷款的情况
        uid_nearest_loan = df_loan_tmp[maxtime_idx].reset_index(drop=True)
        click_loan = uid_nearest_loan[["uid","loan_time"]].merge(df_click[["uid","click_time"]], on="uid", how="left")
        if idx == 0:
            valid_idx = (click_loan["click_time"] > click_loan["loan_time"]) & (click_loan["click_time"] < pd.Timestamp("2016-11-01"))
            afterLoanOrderStatus = click_loan[valid_idx].groupby("uid")["click_time"].count().rename("afterNearestLoanClickCnt").reset_index()
            duser1 = duser1.merge(afterLoanOrderStatus, on="uid", how="left")
            duser1["afterNearestLoanClickCnt"] = duser1["afterNearestLoanClickCnt"].fillna(0)
        elif idx ==1:
            valid_idx = (click_loan["click_time"] > click_loan["loan_time"]) & (click_loan["click_time"] < pd.Timestamp("2016-12-01"))
            afterLoanOrderStatus = click_loan[valid_idx].groupby("uid")["click_time"].count().rename("afterNearestLoanClickCnt").reset_index()
            duser2 = duser2.merge(afterLoanOrderStatus, on="uid", how="left")
            duser2["afterNearestLoanClickCnt"] = duser2["afterNearestLoanClickCnt"].fillna(0)
    return duser1, duser2
    

##每个人最近一次购买之后贷款金额，次数, 
def getLoanStatusAfterNearestOrder(df_loan, df_order, duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(df_order, "buy_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        df_order_tmp = df_order[mask].reset_index(drop=True)
        maxtime_idx = df_order_tmp.groupby(['uid'])['buy_time'].transform(max) == df_order_tmp['buy_time']  #用户最近一次贷款的情况
        uid_nearest_order = df_order_tmp[maxtime_idx][["uid","buy_time"]].drop_duplicates().reset_index(drop=True)
        order_loan = df_loan.merge(uid_nearest_order, on="uid", how="left")
        if idx == 0:
            valid_idx = (order_loan["buy_time"] <= order_loan["loan_time"]) & (order_loan["loan_time"] < pd.Timestamp("2016-11-01"))
            afterLoanOrderStatus = order_loan[valid_idx].groupby("uid")["loan_amount"].agg(["count","sum"]).reset_index()
            afterLoanOrderStatus.columns = ["uid","afterNearestOrderLoanCnt", "afterNearestOrderLoanAmt"]
            duser1 = duser1.merge(afterLoanOrderStatus, on="uid", how="left")
            duser1[["afterNearestOrderLoanCnt", "afterNearestOrderLoanAmt"]] = duser1[["afterNearestOrderLoanCnt", "afterNearestOrderLoanAmt"]].fillna(0)
        elif idx ==1:
            valid_idx = (order_loan["buy_time"] <= order_loan["loan_time"]) & (order_loan["loan_time"] < pd.Timestamp("2016-12-01"))
            afterLoanOrderStatus = order_loan[valid_idx].groupby("uid")["loan_amount"].agg(["count","sum"]).reset_index()
            afterLoanOrderStatus.columns = ["uid","afterNearestOrderLoanCnt", "afterNearestOrderLoanAmt"]
            duser2 = duser2.merge(afterLoanOrderStatus, on="uid", how="left")
            duser2[["afterNearestOrderLoanCnt", "afterNearestOrderLoanAmt"]] = duser2[["afterNearestOrderLoanCnt", "afterNearestOrderLoanAmt"]].fillna(0)
    return duser1, duser2

def gen_fixed_tw_features_for_order_01(df_order, df_loan, duser1, duser2, window_size):
    valid_mask, test_mask = get_windows_mask(df_order, "buy_time", window_size)
    loan_valid_mask, loan_test_mask = get_windows_mask(df_loan, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df_order[mask].reset_index(drop=True)
        tmp['is_discount']=(tmp.discount!=0.0) * 1.0
        if idx==0:
            tmp['daysOrder']=(pd.Timestamp(valid_end_date)-tmp.buy_time).apply(lambda x:x.days+x.seconds/86400.0)
            loan_uid = df_loan[loan_valid_mask].groupby('uid').loan_amount.sum().rename('loan_amt').reset_index()
        elif idx==1:
            tmp['daysOrder']=(pd.Timestamp(test_end_date)-tmp.buy_time).apply(lambda x:x.days+x.seconds/86400.0)
            loan_uid = df_loan[loan_test_mask].groupby('uid').loan_amount.sum().rename('loan_amt').reset_index() 
        code_cate_id = (tmp[['uid','cate_id']].drop_duplicates().merge(loan_uid, how="left", on="uid").groupby('cate_id').loan_amt.sum()/loan_uid.loan_amt.sum()).rename('code_cate_id').reset_index()
        tmp = tmp.merge(code_cate_id, how="left", on='cate_id')
        stat_codeCateId = tmp.groupby('uid').code_cate_id.agg(['sum','mean','max','min']).reset_index()
        stat_codeCateId.columns=['uid']+[i+'_codeCateId_'+str(window_size) for i in list(stat_codeCateId.columns)[1:]]
        #购买类别
        stat_codeId = tmp.groupby('uid').cate_id.agg('nunique').reset_index()
        stat_codeId.columns = ['uid']+['nunique'+'_CateId_'+str(window_size) for i in list(stat_codeId.columns)[1:]]
        #购买数量的sum
        stat_qty=tmp.groupby('uid').qty.agg('sum').reset_index()
        stat_qty.columns = ['uid']+['sum'+'_qty_'+str(window_size) for i in list(stat_qty.columns)[1:]]
        #金额最大的一次的code_cate_id和order_amt
        order_amt_idxmax = tmp.groupby('uid').order_amt.idxmax()
        stat_order_amt_max = tmp.loc[list(order_amt_idxmax)][['uid','code_cate_id','order_amt']]
        stat_order_amt_max.columns = ['uid']+[i+'_maxAmtOrder_'+str(window_size) for i in list(stat_order_amt_max.columns)[1:]]
        #最近的一次的code_cate_id和order_amt
        stat_daysOrder = tmp.groupby('uid').daysOrder.agg(['mean','max','min']).reset_index()
        stat_daysOrder.columns = ['uid']+[i+'_daysOrder_'+str(window_size) for i in list(stat_daysOrder.columns)[1:]]
        cur_order=tmp.groupby('uid').daysOrder.idxmin()
        stat_cur_order = tmp.loc[list(cur_order)][['uid','code_cate_id','order_amt','daysOrder']]
        stat_cur_order.columns = ['uid']+[i+'_curOrder_'+str(window_size) for i in list(stat_cur_order.columns)[1:]]
        #使用的discount的次数和占比
        stat_isDiscount = tmp.groupby('uid').is_discount.sum().reset_index()
        stat_isDiscount.columns = ['uid','cnt_discount_'+str(window_size)]
        stat_Discount = tmp.groupby('uid')['discount','order_amt'].sum().reset_index()
        stat_Discount['discount_ratio'] = stat_Discount.discount/(stat_Discount.discount+stat_Discount.order_amt)
        stat_Discount.columns = ['uid','sum_discount_'+str(window_size),'sum_orderAmt_'+str(window_size),'discount_ratio_'+str(window_size)]
        stat = stat_codeCateId.merge(stat_codeId, how="left", on="uid").merge(stat_qty, how="left", on="uid").merge(stat_order_amt_max, how="left", on="uid").merge(stat_daysOrder, how="left", on="uid").merge(stat_cur_order, how="left", on="uid").merge(stat_isDiscount, how="left", on="uid").merge(stat_Discount[['uid','discount_ratio_'+str(window_size)]], how="left", on="uid")
        if idx==0:
            duser1=duser1.merge(stat, how="left", on="uid")
        elif idx==1:    
            duser2=duser2.merge(stat, how="left", on="uid")       
    return duser1,duser2  


##下次贷款金额比上间隔
tr_user, ts_user = getLoanDaily(tr_user, ts_user)

def getLoanDaily(duser1, duser2):
    valid_mask = t_loan.month.isin([8,9,10])
    test_mask = t_loan.month.isin([9,10,11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = t_loan[mask].reset_index()
        tmp["dailyLoanAmt"] = tmp["loan_amount"]/ (tmp["loan_interval"]+1)
        meanDailyLoanAmt = tmp.groupby("uid")["dailyLoanAmt"].agg(["mean","max","min"]).reset_index()
        if idx == 0:
            duser1 = duser1.merge(meanDailyLoanAmt, how="left", on = 'uid')
        elif idx == 1:
            duser2 = duser2.merge(meanDailyLoanAmt, how="left", on = 'uid')
    return duser1, duser2

tr_user, ts_user = getCatePivotAmtCnt(t_order, tr_user, ts_user)
#类目消费金额的差分
def getCatePivotAmtCnt(df, duser1, duser2):
    valid_mask = t_order.month.isin([8, 9,10])
    test_mask = t_order.month.isin([9, 10,11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = t_order[mask]
        uid_months = tmp.groupby(["uid","month","cate_id"])["order_amt"].agg(["sum"]).reset_index()
        uid_months.rename({'sum': 'cate_id_sum' }, axis='columns',inplace=True)
        uid_months["cate_id"]  = "cate_id_" + uid_months['cate_id'].astype(str)
        cols = ["uid","cate_id","cate_id_cnt","cate_id_sum"]
        months_cate = pd.DataFrame()
        for i in uid_months["month"].unique():
            tmp = uid_months.loc[uid_months["month"]==i,cols].reset_index(drop=True)
            uid_months_cate = tmp.pivot(index='uid', columns='cate_id').reset_index().fillna(0)
            new_list = ["uid"]
            for words in uid_months_cate.columns.get_values():
                if "uid" in words:
                    continue
                new_list.append('_'.join(words))
            uid_months_cate.columns =  new_list
            uid_months_cate["month"] = i
            months_cate = pd.concat([months_cate,uid_months_cate])
        months_cate1 = months_cate.pivot(index='uid', columns='month').fillna(0)
        months_cate1 = months_cate1.stack().reset_index()
        months_cate1 = months_cate1.groupby(["uid"]).apply(lambda x: x.sort_values(["month"], ascending=True)).reset_index(drop=True)
        fea = set(months_cate1.columns) - set(["uid","month"])
        for i in fea:
            months_cate1[i + "_Diff"] = months_cate1.groupby("uid")[i].apply(lambda x: x - x.shift(1))
            #months_cate1[i + "_Diff2"] = months_cate1.groupby("uid")[i + "_Diff"].apply(lambda x: x - x.shift(1))
        col = ["uid"]
        for i in months_cate1.columns:
            if "Diff" in i:
                col.extend([i])
        print col
        if idx == 0:
            duser1 = duser1.merge(months_cate1[months_cate1["month"] == 10, col], how="left", on="uid")
        elif idx == 1:
            duser2 = duser2.merge(months_cate1[months_cate1["month"] == 11, col], how="left", on="uid")
    return duser1, duser2
#贷款序列编码
tr_user, ts_user = codeLoanSeq(t_loan, tr_user, ts_user)

def codeLoanSeq(df_loan, duser1, duser2):
    valid_month = 11
    test_month = 12
    for idx, s_month in enumerate(range(valid_month-3,valid_month)):
        monthmask = df_loan.month.isin([s_month])
        uid_tmp = df_loan[monthmask].uid.unique()
        df_loan["isloan" + str(3-idx)+"monthago"] = df_loan.uid.isin(uid_tmp) * 1
    df_loan["loanSeq"] = df_loan["isloan3monthago"].astype(str)+df_loan["isloan2monthago"].astype(str)+df_loan["isloan1monthago"].astype(str)
    duser1 = duser1.merge(df_loan[["uid","loanSeq"]], on="uid", how="left")
    duser1["loanSeq"] = duser1["loanSeq"].fillna("000")
    #测试集
    for idx, s_month in enumerate(range(test_month-3,test_month)):
        monthmask = df_loan.month.isin([s_month])
        uid_tmp = df_loan[monthmask].uid.unique()
        df_loan["isloan" + str(3-idx)+"monthago"] = df_loan.uid.isin(uid_tmp) * 1
    df_loan["loanSeq"] = df_loan["isloan3monthago"].astype(str)+df_loan["isloan2monthago"].astype(str)+df_loan["isloan1monthago"].astype(str)
    duser2 = duser2.merge(df_loan[["uid","loanSeq"]], on="uid", how="left")
    duser2["loanSeq"] = duser2["loanSeq"].fillna("000")
    return duser1, duser2

tr_user["loanSeq"] = tr_user["loanSeq"].astype('category')
tr_user['loanSeq'].cat.categories=range(tr_user["loanSeq"].nunique())  #148
tr_user["loanSeq"] = tr_user["loanSeq"].astype(int)

ts_user["loanSeq"] = ts_user["loanSeq"].astype('category')
ts_user['loanSeq'].cat.categories=range(ts_user["loanSeq"].nunique())  #148
ts_user["loanSeq"] = ts_user["loanSeq"].astype(int)



tr_user, ts_user = getMaxLoanAmt(tr_user, ts_user, 60)
tr_user, ts_user = getMaxLoanAmt(tr_user, ts_user, 90)


def getMaxLoanAmt(duser, tuser, window_size):
    valid_mask, test_mask = get_windows_mask(t_loan, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = t_loan[mask]
        tmp = tmp.groupby(['uid','date'])["loan_amount"].sum().rename("dailyLoanAmt").reset_index()  #每个人一天内贷款了多少金额
        tmp["maxDailyLoanAmt"] = tmp.groupby(['uid'])['dailyLoanAmt'].transform(max)  #用户消费最大的情况
        uid_max_loan = tmp[(tmp["dailyLoanAmt"] == tmp["maxDailyLoanAmt"])]
        max_amt_idx = uid_max_loan.groupby("uid")['date'].transform(max) == uid_max_loan['date']
        uid_max_loan = uid_max_loan[max_amt_idx]
        if idx == 0:
            uid_max_loan["maxLoan_interval"] = (pd.Timestamp(valid_end_date) - uid_max_loan["date"]).apply(lambda x:x.days)
        elif idx == 1:
            uid_max_loan["maxLoan_interval"] = (pd.Timestamp(test_end_date) - uid_max_loan["date"]).apply(lambda x:x.days) 
        uid_max_loan["maxloan_price_interval"] = uid_max_loan["maxDailyLoanAmt"] / (uid_max_loan["maxLoan_interval"]+1)
        uid_max_loan.drop(["dailyLoanAmt","date"],axis=1,inplace=True)
        if idx == 0:
            duser = duser.merge(uid_max_loan,how="left",on="uid")
        elif idx == 1:
            tuser = tuser.merge(uid_max_loan,how="left",on="uid")
    return duser, tuser


tr_user, ts_user = getAgeLoan(t_loan, tr_user, ts_user)


tr_user, ts_user =  getFixedSexLoan(t_loan, tr_user, ts_user, 30)
###从sex的角度####
def getFixedSexLoan(df,duser1,duser2,window_size):
    valid_mask, test_mask = get_windows_mask(df, "loan_time", window_size)
    for idx, mask in enumerate([valid_mask, test_mask]):
         tmp = df[mask].reset_index(drop=True)
         tmp = tmp.merge(t_user[["uid","sex"]], on ="uid", how="left")
         stat_loanAmt = tmp.groupby(["sex"])['loan_amount'].agg(['sum','mean','count','median']).reset_index()
         stat_loanAmt.columns=['sex']+ [i + '_sex'+'_loanAmt_'+str(window_size)+"d" for i in list(stat_loanAmt.columns)[1:]]
         if idx==0:
             duser1=duser1.merge(stat_loanAmt, how="left", on="sex")
         elif idx==1:
             duser2=duser2.merge(stat_loanAmt, how="left", on="sex")
    return duser1, duser2


###从age的角度的贷款金额
def getAgeLoan(df, duser1, duser2):
    valid_mask = df.month.isin([10])
    test_mask = df.month.isin([11])
    for idx, mask in enumerate([valid_mask, test_mask]):
        tmp = df[mask].reset_index(drop=True)
        tmp = tmp.merge(t_user[["uid","age"]], on ="uid", how="left")
        stat_loanAmt = tmp.groupby(["age"])['loan_amount'].agg(['sum','count','mean','median']).reset_index()
        stat_loanAmt.columns=['age']+ [i+ '_AgeLoanAmt' for i in list(stat_loanAmt.columns)[1:]]
        if idx==0:
            duser1 = duser1.merge(stat_loanAmt, how="left", on="age")
        elif idx==1:
            duser2 = duser2.merge(stat_loanAmt, how="left", on="age")     
    return duser1, duser2


##购买占比limit
tr_user, ts_user = getOrderAmtLimit(tr_user, ts_user)

def getOrderAmtLimit(duser1, duser2):
    valid_mask = t_order.month == 10
    test_mask = t_order.month == 11
    for idx, mask in enumerate([valid_mask, test_mask]):
        lastMonthOrderAmt = t_order[mask].groupby("uid")["order_amt"].sum().rename("lastMonthOrderAmt").reset_index()
        if idx == 0:
           duser1 = duser1.merge(lastMonthOrderAmt, on="uid", how="left")
           duser1["lastMonthOrderAmt_Limit"] = duser1["lastMonthOrderAmt"] /duser1["limit"]
           duser1.drop("lastMonthOrderAmt",inplace=True,axis=1)
        elif idx == 1:
           duser2 = duser2.merge(lastMonthOrderAmt, on="uid", how="left")
           duser2["lastMonthOrderAmt_Limit"] = duser2["lastMonthOrderAmt"] /duser2["limit"]
           duser2.drop("lastMonthOrderAmt",inplace=True,axis=1)
    return duser1, duser2



##最近N的点击记录序列

tr_user, ts_user = getNthNearestClick(tr_user, ts_user)

def getNthNearestClick(duser1, duser2):
    click_validmask = t_click.month < 11
    click_testmask = t_click.month < 12
    for idx, mask in enumerate([click_validmask, click_testmask]):
        t_click_tmp = t_click[mask].reset_index(drop=True)
        t_click_tmp["pid_param"]= t_click_tmp["pid"].astype(str) + "_" + t_click_tmp["param"].astype(str)
        t_click_tmp["loan_rank"] = t_click_tmp.groupby(["uid"])["click_time"].rank(ascending=False)
        for pos in [1,2,3,4,5]:
            t_click_tmp1 = t_click_tmp[t_click_tmp["loan_rank"] == pos][["uid","pid_param"]]
            t_click_tmp1.columns = ["uid", "pid_param_pos" + str(pos)]
            if idx ==0:
                duser1 = duser1.merge(t_click_tmp1, on="uid", how="left")
            elif idx == 1:
                duser2 = duser2.merge(t_click_tmp1, on="uid", how="left")
        dropcol= ["pid_param_pos" + str(1), "pid_param_pos" + str(2), "pid_param_pos" + str(3), "pid_param_pos" + str(4), "pid_param_pos" + str(5)]
        if idx ==0:
            duser1["pid_param_pos"] = duser1["pid_param_pos" + str(1)] + duser1["pid_param_pos" + str(2)] + duser1["pid_param_pos" + str(3)] + duser1["pid_param_pos" + str(4)] + duser1["pid_param_pos" + str(5)]
            duser1["pid_param_pos"] = duser1["pid_param_pos"].astype('category')
            duser1["pid_param_pos"].cat.categories= np.arange(1,duser1["pid_param_pos"].nunique()+1)
            duser1["pid_param_pos"] = duser1["pid_param_pos"].astype(int)
            duser1.drop(dropcol, axis=1 ,inplace=True)
        elif idx == 1:
            duser2["pid_param_pos"] = duser2["pid_param_pos" + str(1)] + duser2["pid_param_pos" + str(2)] + duser2["pid_param_pos" + str(3)] + duser2["pid_param_pos" + str(4)] + duser2["pid_param_pos" + str(5)]
            duser2["pid_param_pos"] = duser2["pid_param_pos"].astype('category')
            duser2["pid_param_pos"].cat.categories= np.arange(1,duser2["pid_param_pos"].nunique()+1)
            duser2["pid_param_pos"] = duser2["pid_param_pos"].astype(int)
            duser2.drop(dropcol, axis=1 ,inplace=True)
    return duser1, duser2



def get_ordernum_window(df, gkey, window_size):
	grouped = df.groupby(gkey)['order_id', 'order_unix_time']
	order_num = grouped.rolling(window = window_size, on = 'order_unix_time', closed = 'left').count().reset_index().fillna(0)
	df = df.merge(order_num[["level_1",newcol]], how='left', left_index=True, right_on='level_1').drop(['level_1'], axis = 1)
	return df



selected_mask= tr_user.loan_sum<=7.5

selected_tr_user = tr_user[selected_mask]

select_rows1 = random.sample(selected_tr_user.index, int(len(selected_tr_user.index)*0.7))
selected_train_df = selected_tr_user.loc[select_rows1]
selected_valid_df = selected_tr_user.drop(select_rows1)

selected_dtrain = lgb.Dataset(selected_train_df[features], label=selected_train_df["loan_sum"], free_raw_data=False)
selected_dvalid = lgb.Dataset(selected_valid_df[features], label=selected_valid_df["loan_sum"], free_raw_data=False)


#"feature_fraction":0.66, "bagging_freq" : 1 , "bagging_fraction": 0.6  ,'lambda_l2':0.0
param = {'num_leaves':8,'num_boost_round':300, 'objective':'regression_l2','metric':'rmse',"learning_rate" : 0.05, "boosting":"gbdt"} 
bst = lgb.train(param, selected_dtrain,  verbose_eval=100)
pred_lgb_train = bst.predict(selected_dtrain.data)
pred_lgb_valid = bst.predict(selected_dvalid.data)
print('train mae: %g' % sqrt(mean_squared_error(selected_train_df["loan_sum"], pred_lgb_train)))
valid_score = sqrt(mean_squared_error(selected_valid_df["loan_sum"], pred_lgb_valid))
print('valid mae: %g' % valid_score)


##后向选择算法
logfeatures = list(imp[imp[1] != 0][0])

bestre = 1.77239
removes =[]
for col in set(logfeatures):
    print "removing: ", col
    features.remove(col)
    dtrain = lgb.Dataset(train_df[features], label=train_df["loan_sum"],free_raw_data=False)
    dvalid = lgb.Dataset(valid_df[features], label=valid_df["loan_sum"], free_raw_data=False)
    dtrain_all = lgb.Dataset(tr_user[features], label=tr_user["loan_sum"], free_raw_data=False)
    dtest = lgb.Dataset(ts_user[features], free_raw_data=False)

    param = {'num_leaves':8,'num_boost_round':500, 'objective':'regression_l2','metric':'rmse',"learning_rate" : 0.05, "boosting":"gbdt", "lambda_l2":1500, "feature_fraction":0.9, "bagging_fraction":0.9, "bagging_freq" : 50} 
    bst = lgb.train(param, dtrain, valid_sets=[dtrain, dvalid],  verbose_eval=100)
    pred_lgb_train = bst.predict(dtrain.data)
    pred_lgb_valid = bst.predict(dvalid.data)
    print('train mae: %g' % sqrt(mean_squared_error(train_df["loan_sum"], pred_lgb_train)))
    valid_score = sqrt(mean_squared_error(valid_df["loan_sum"], pred_lgb_valid))
    print('valid mae: %g' % valid_score)
    if bestre > valid_score:
        bestre = valid_score
        print "removed: ", col
        removes.extend([col])
    else:
        features.extend([col])



