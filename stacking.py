from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from vecstack import stacking
from math import sqrt

import numpy as np
import scipy.stats as st
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def transformer(y, func=None):
    if func is None:
        return y
    else:
        return func(y)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def stacking(models, X_train, y_train, X_test, regression=True,
             transform_target=None, transform_pred=None,
             metric=None, n_folds=4, stratified=False,
             shuffle=False, random_state=0, verbose=0):
    # Print type of task
    if regression and verbose > 0:
        print('task:   [regression]')
    elif not regression and verbose > 0:
        print('task:   [classification]')

    # Specify default metric for cross-validation
    if metric is None and regression:
        metric = mean_absolute_error
    elif metric is None and not regression:
        metric = accuracy_score
        
    # Print metric
    if verbose > 0:
        print('metric: [%s]\n' % metric.__name__)
        
    # Split indices to get folds (stratified can be used only for classification)
    if stratified and not regression:
        kf = StratifiedKFold(y_train, n_folds, shuffle = shuffle, random_state = random_state)
    else:
        kf = KFold(len(y_train), n_folds, shuffle = shuffle, random_state = random_state)

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))
    
    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('model %d: [%s]' % (model_counter, model.__class__.__name__))
            
        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((X_test.shape[0], len(kf)))
        
        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf):
            X_tr = X_train.loc[tr_index]
            y_tr = y_train.loc[tr_index]
            X_te = X_train.loc[te_index]
            y_te = y_train.loc[te_index]
            
            # Fit 1-st level model
            model = model.fit(X_tr, transformer(y_tr, func = transform_target))
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = transformer(model.predict(X_te), func = transform_pred)
            # Predict full test set
            S_test_temp[:, fold_counter] = transformer(model.predict(X_test), func = transform_pred)
            
            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, metric(y_te, S_train[te_index, model_counter])))
                
        # Compute mean or mode of predictions for test set
        if regression:
            S_test[:, model_counter] = np.mean(S_test_temp, axis = 1)
        else:
            S_test[:, model_counter] = st.mode(S_test_temp, axis = 1)[0].ravel()
            
        if verbose > 0:
            print('    ----')
            print('    MEAN:   [%.8f]\n' % (metric(y_train, S_train[:, model_counter])))

    return (S_train, S_test)



X_train = tr_user[features].replace([np.inf,np.nan], 0).reset_index(drop=True)
X_test = ts_user[features].replace([np.inf,np.nan], 0).reset_index(drop=True)
y_train = tr_user["loan_sum"].reset_index(drop=True)


# Caution! All models and parameter values are just 
# demonstrational and shouldn't be considered as recommended.
# Initialize 1-st level models.
models = [
    ExtraTreesRegressor(random_state = 0, n_jobs = -1, 
        n_estimators = 300, max_depth = 3),
        
    RandomForestRegressor(random_state = 0, n_jobs = -1, 
        n_estimators = 300, max_depth = 3),
        
    XGBRegressor(seed = 0, learning_rate = 0.05, 
        n_estimators = 300, max_depth = 3),

    LGBMRegressor(num_leaves = 8, learning_rate = 0.05, n_estimators= 300)
    ]
    
# Compute stacking features

S_train, S_test = stacking(models, X_train, y_train, X_test, regression = True, metric = mean_squared_error, n_folds = 5, shuffle = True, random_state = 0, verbose = 2)

  
# Fit 2-nd level model
model =  LGBMRegressor(num_leaves = 8, learning_rate = 0.05, n_estimators= 300)
model = model.fit(S_train, y_train)
y_pred = model.predict(S_test)

id_test = ts_user['uid']
stacking_sub = pd.DataFrame({'uid': id_test, 'stacking_loan_sum': y_pred})
print(stacking_sub.describe())
stacking_sub.loc[stacking_sub["stacking_loan_sum"] < 0,"stacking_loan_sum"] = 0
print('saving submission...')
now_time = time.strftime("%m-%d %H_%M_%S", time.localtime()) 
stacking_sub[["uid","stacking_loan_sum"]].to_csv("./submission/" +now_time+'_stacking.csv', index=False, header=False)


