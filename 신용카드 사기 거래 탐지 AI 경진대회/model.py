# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:07:48 2022

@author: user
"""


import numpy as np
# Basic Libraries
import pandas as pd
import numpy as np

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


#
train_df = pd.read_csv('train.csv')


val_df = pd.read_csv('val.csv')


test_df = pd.read_csv('test.csv')

sample_sub = pd.read_csv('sample_submission.csv')





# %% [markdown]
# # Validation 분류

# %%
# Validation == 1 (사기거래)에 해당하는 값 추출
val_df01 = val_df[(val_df['Class'] == 1)].drop(['ID', 'Class'], axis = 1)

# Validation == 0 (정상거래)에 해당하는 값 추출
val_df00 = val_df[(val_df['Class'] == 0)].drop(['ID', 'Class'], axis = 1)

# %%
val_df01_feature =val_df01[['V2','V3','V4','V7','V9','V10','V11','V12','V14','V16','V17','V18']]

# %%
# 사기거래 유형의 각 피처별 평균값 data 생성
fraud = pd.DataFrame(val_df01_feature.mean()).transpose()
fraud = fraud.to_numpy()
fraud

# %% [markdown]
# # Cosine Similarity

# %%
# ID 컬럼을 제거한 train set을 배열 형태로 변환
train_x = train_df.drop(columns=['ID']) 
train_x = train_x[['V2','V3','V4','V7','V9','V10','V11','V12','V14','V16','V17','V18']]
train_x = train_x.to_numpy()

train_x

# %%
# 코사인 유사도 함수 생성

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


from scipy.spatial.distance import cdist

# %%
# for문을 활용하여 코사인유사도 값 할당
credit = []

for i in range(len(train_df)):
    credit.append(cosine_similarity(train_x[i], fraud[0]))


# for i in range(len(train_df)):
#     c_distance=cdist(train_x[i], fraud[0],metric='mahalanobis')
#     credit.append(np.diag(c_distance))



# %%
train_df['cosine'] = credit
# train_df['cosine'] = abs(train_df['cosine'])
train_df

# %%
train_df['Class'] = np.where(train_df['cosine'] > 0.7, 1, 0)
train_df['cosine'].value_counts()





#train set fraud labeling
fraud = val_df[val_df['Class']==1]

fraud['V14'].describe()

#boxplot 75%


detect_fraud_2 = fraud['V2'].describe()[4]
detect_fraud_3 = fraud['V3'].describe()[6]
detect_fraud_4 = fraud['V4'].describe()[4]
detect_fraud_7 = fraud['V7'].describe()[6]
detect_fraud_9 = fraud['V9'].describe()[6]
detect_fraud_10 = fraud['V10'].describe()[6]
detect_fraud_11 = fraud['V11'].describe()[4]
detect_fraud_12 = fraud['V12'].describe()[6]
detect_fraud_14 = fraud['V14'].describe()[6]
detect_fraud_16 = fraud['V16'].describe()[6]
detect_fraud_17 = fraud['V17'].describe()[6]
detect_fraud_18 = fraud['V18'].describe()[6]

train_df['Class'] = None



#train set labeling

count_list = []
len(count_list)

for i in range(len(train_df)) :
    count = 0
    if train_df['V2'][i] > detect_fraud_2:
        count += 1
    if train_df['V4'][i] > detect_fraud_4:
        count += 1
    if train_df['V11'][i] > detect_fraud_11:
        count += 1
    
    if train_df['V3'][i] < detect_fraud_3:
        count += 1
    if train_df['V7'][i] < detect_fraud_7:
        count += 1
    if train_df['V9'][i] < detect_fraud_9:
        count += 1    
    if train_df['V10'][i] < detect_fraud_10:
        count += 1
    if train_df['V12'][i] < detect_fraud_12:
        count += 1
    if train_df['V14'][i] < detect_fraud_14:
        count += 1
    if train_df['V16'][i] < detect_fraud_16:
        count += 1    
    if train_df['V17'][i] < detect_fraud_17:
        count += 1
    if train_df['V18'][i] < detect_fraud_18:
        count += 1
           
    if train_df['cosine'][i] > 0.7 :
        count +=1
        
    if count > 6 :
        train_df["Class"][i] = 1
        count_list.append(train_df["ID"][i])
        
    else:
        train_df['Class'][i] =0
        print(i)



for i in range(len(train)) :
    count = 0

    if train['V14'][i] < detect_fraud_14:
        count += 1
    if train['V17'][i] < detect_fraud_17:
        count += 1
        
    if count == 2 :
        train["Class"][i] = 1
        count_list.append(train["ID"][i])
        
    else:
        train['Class'][i] =0
        print(i)


train_df.to_csv('new_train_boxplot_75_7.csv',index=False)




import pandas as pd
train = pd.read_csv('new_train_boxplot_75_7.csv')


val = pd.read_csv('val.csv')


test = pd.read_csv('test.csv')

sample_sub = pd.read_csv('sample_submission.csv')



x_val = val.drop(columns=['ID', 'Class']) # Input Data
y_val = val['Class'] # Label





x_train = train.drop(['ID','Class', 'cosine'],axis=1)

y_train = train['Class']



x_test = test.drop(['ID'],axis=1)


# y_train.value_counts()


# from imblearn.under_sampling import RandomUnderSampler


# under_sampling_smote = RandomUnderSampler(sampling_strategy='majority')



from imblearn.over_sampling import SMOTE


#x_train_over, y_train_over = SMOTE().fit_resample(x_train, y_train)


#class와 상관관계 0.5 이상
#x_train = train[['V2','V3','V4','V7','V9','V10','V11','V12','V14','V16','V17','V18']]



#x_train_under , y_train_under = under_sampling_smote.fit_resample(x_train,y_train)



from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

RANDOM_STATE=123
#best score
# x_train = train[['V2','V3','V4','V7','V9','V10','V11','V12','V14','V16','V17','V18']]

# VERBOSE_EVAL = 50 #Print out metric result
# clf = CatBoostClassifier(iterations=100,
#                               learning_rate=0.02,
#                               depth=12,
#                               eval_metric='AUC',
#                               random_seed = RANDOM_STATE,
#                               bagging_temperature = 0.9,
#                               od_type='Iter',
#                               metric_period = VERBOSE_EVAL,
#                               od_wait=100)



VERBOSE_EVAL = 50 #Print out metric result
clf = CatBoostClassifier()


#clf.fit(x_train_over, y_train_over)
clf.fit(x_train, y_train)

y_pred =clf.predict(x_val)

# val_pred = get_pred_label(val_pred)
val_score = f1_score(y_val, y_pred, average='macro')

print(f'Validation F1 Score : [{val_score}]')
print(classification_report(y_val, y_pred))


test_pred = clf.predict(x_test )

np.count_nonzero(test_pred)


sample_sub['Class'] = test_pred

sample_sub.to_csv('cosine07_no_oversampling_count7.csv',index=False)











import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


val_df = val.drop(["ID", "Class"], axis = 1)

corr = val_df.corr(method= "pearson") 



















#hyperparameter tuning
import optuna

from sklearn.metrics import roc_auc_score

#catboost

def objective(trial, data=x_train, target=y_train):
    params = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        'max_depth': trial.suggest_int('max_depth', 3, 16),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.02, 0.05, 0.08, 0.1]),
        # 'n_estimators': trial.suggest_int('n_estimators', 2000, 8000),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 1.0, log = True),
        'subsample': trial.suggest_float('subsample', 0.1, 0.8),
        'iterations' : trial.suggest_int('iterations',100,500),
        'random_seed': 123,
        'bagging_temperature' : trial.suggest_float('bagging_temperature',0.2,0.9),
        # 'task_type': 'GPU',
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        # 'bootstrap_type': 'Poisson'
    }
    
    model = CatBoostClassifier(**params)  
    model.fit(x_train, y_train, eval_set = [(x_val,y_val)], early_stopping_rounds = 222, verbose = False)
    y_pred = model.predict_proba(x_val)[:,1]
    roc_auc = roc_auc_score(y_val, y_pred)

    return roc_auc


study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 20)
print('Best value:', study.best_value)




params = study.best_params



RANDOM_STATE=123

VERBOSE_EVAL = 50 #Print out metric result
clf = CatBoostClassifier(**params)

clf.fit(x_train,y_train)

y_pred =clf.predict(x_val)



# val_pred = get_pred_label(val_pred)
val_score = f1_score(y_val, y_pred, average='macro')

print(f'Validation F1 Score : [{val_score}]')
print(classification_report(y_val, y_pred))


test_pred = clf.predict(x_test)

sample_sub['Class'] = test_pred

sample_sub.to_csv('C:/Users/user/Desktop/대학원수업/dacon/credict_card/catboost_optuna_origin_labeling.csv',index=False)




#xgboost

from xgboost import XGBClassifier


def objective(trial, data=x_train, target=y_train):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 32),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.02, 0.05, 0.08, 0.1]),
        'n_estimators': trial.suggest_int('n_estimators', 2000, 8000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma': trial.suggest_float('gamma', 0.0001, 1.0, log = True),
        'alpha': trial.suggest_float('alpha', 0.0001, 10.0, log = True),
        'lambda': trial.suggest_float('lambda', 0.0001, 10.0, log = True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
        'subsample': trial.suggest_float('subsample', 0.1, 0.8),
        # 'tree_method': 'gpu_hist',
        'booster': 'gbtree',
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'

    }
    
    model = XGBClassifier(**params)  
    model.fit(x_train, y_train, eval_set = [(x_val,y_val)], early_stopping_rounds = 333, verbose = False)
    y_pred = model.predict_proba(x_val)[:,1]
    roc_auc = roc_auc_score(y_val, y_pred)

    return roc_auc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print('Best value: ', study.best_value)

params = study.best_params

clf =XGBClassifier(**params)  

clf.fit(x_train,y_train)

y_pred =clf.predict(x_val)



# val_pred = get_pred_label(val_pred)
val_score = f1_score(y_val, y_pred, average='macro')

print(f'Validation F1 Score : [{val_score}]')
print(classification_report(y_val, y_pred))


test_pred = clf.predict(x_test)

sample_sub['Class'] = test_pred

sample_sub.to_csv('C:/Users/user/Desktop/대학원수업/dacon/credict_card/lgbm_origin_labeling.csv',index=False)


#ligth gbm

from lightgbm import LGBMClassifier
def objective(trial,data=x_train,target=y_train):   
    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 11, 333),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 64),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.005, 0.1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 2000, 8000),
        'cat_smooth' : trial.suggest_int('cat_smooth', 10, 100),
        'cat_l2': trial.suggest_int('cat_l2', 1, 20),
        'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
        'cat_feature' : [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 
                         53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
        'n_jobs' : -1, 
        'random_state': 42,
        'boosting_type': 'gbdt',
        'metric': 'AUC',
        # 'device': 'gpu'
    }
    model = LGBMClassifier(**params)  
    model.fit(x_train,y_train,eval_set=[(x_val,y_val)],eval_metric='auc', early_stopping_rounds=300, verbose=False)
    preds = model.predict_proba(x_val)[:,1]
    auc = roc_auc_score(y_val, preds)
    
    return auc



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)



params = study.best_params

clf =LGBMClassifier(**params)  

clf.fit(x_train,y_train)

y_pred =clf.predict(x_val)



# val_pred = get_pred_label(val_pred)
val_score = f1_score(y_val, y_pred, average='macro')

print(f'Validation F1 Score : [{val_score}]')
print(classification_report(y_val, y_pred))


test_pred = clf.predict(x_test)

sample_sub['Class'] = test_pred

sample_sub.to_csv('C:/Users/user/Desktop/대학원수업/dacon/credict_card/lgbm_origin_labeling.csv',index=False)