import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import matplotlib.pyplot as plt 



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
age = pd.read_csv("age_gender_info.csv")



##데이터확인

train.info()



#칼럼명 변셩
train.columns = [
    '단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '신분',
    '임대보증금', '임대료', '지하철', '버스',
    '단지내주차면수', '등록차량수'
]

test.columns = [
    '단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '신분',
    '임대보증금', '임대료', '지하철', '버스',
    '단지내주차면수'
]

#결측치 확인
train.isna().sum()

train.describe()

train.columns



"""
#임대보증금,임대료 제거
train = train.drop(["단지코드","임대보증금","임대료"], axis=1)

test = test.drop(["단지코드","임대보증금","임대료"], axis=1)

corr = train.corr()

train.isna().sum()
test.isna().sum()
"""

#
#임대보증금 float형으로 변경
train['임대보증금']=train['임대보증금'].replace("-",0)
test['임대보증금']=test['임대보증금'].replace("-",0)


train['임대보증금']=train['임대보증금'].astype('float')
test['임대보증금']=test['임대보증금'].astype('float')


#임대료 float형으로 변경
train['임대료']=train['임대료'].replace("-",0)
test['임대료']=test['임대료'].replace("-",0)

train['임대료']=train['임대료'].astype('float')
test['임대료']=test['임대료'].astype('float')


#결측치처리
train = train.fillna(0)
test.loc[(test.신분.isnull()) & (test.단지코드 == "C2411"), '신분'] = 'A'
test.loc[(test.신분.isnull()) & (test.단지코드 == "C2253"), '신분'] = 'C'
test["지하철"] = test["지하철"].fillna(0)

####데이터 준비
"""
age.columns

age = age.drop(['10대미만(여자)', '10대미만(남자)', '10대(여자)', '10대(남자)'],axis=1)
"""
##데이터 축약
#train

group_train = train.groupby(train.단지코드).mean()


group_train["임대건물구분"] = ""
group_train["지역"] = ""
group_train["공급유형"] = ""
group_train["신분"] = ""

for i in range(len(train)):
    for j in range(len(group_train)):
        if train["단지코드"][i]== group_train.index[j]:
            group_train["임대건물구분"][j] = train["임대건물구분"][i]
            group_train["지역"][j] = train["지역"][i]
            group_train["공급유형"][j] = train["공급유형"][i]
            group_train["신분"][j] = train["신분"][i]


group_train_index = group_train.index

"""
group_train = pd.merge(group_train, age, how = 'inner', on="지역")
"""

#test
group_test = test.groupby(test.단지코드).mean()


group_test["임대건물구분"] = ""
group_test["지역"] = ""
group_test["공급유형"] = ""
group_test["신분"] = ""

for i in range(len(test)):
    for j in range(len(group_test)):
        if test["단지코드"][i]== group_test.index[j]:
            group_test["임대건물구분"][j] = test["임대건물구분"][i]
            group_test["지역"][j] = test["지역"][i]
            group_test["공급유형"][j] = test["공급유형"][i]
            group_test["신분"][j] = test["신분"][i]

group_test_index = group_test.index

"""
group_test = pd.merge(group_test, age, how = 'inner', on="지역")
"""

###범주형 더미변수화
##임대건물구분
building_group_train_dummy = pd.get_dummies(group_train["임대건물구분"])
building_group_test_dummy = pd.get_dummies(group_test["임대건물구분"])

building_group_train_dummy.columns == building_group_test_dummy.columns 

group_train = group_train.join(building_group_train_dummy)
group_train = group_train.drop(["임대건물구분"], axis = 1)

group_test = group_test.join(building_group_test_dummy)
group_test = group_test.drop(["임대건물구분"], axis = 1)

##지역
area_group_train_dummy = pd.get_dummies(group_train["지역"])
area_group_test_dummy = pd.get_dummies(group_test["지역"])

#컬럼비교
area_group_train_dummy.columns  
area_group_test_dummy.columns 

#group_test에 없는 컬럼 채우기
area_group_test_dummy["서울특별시"] = 0
area_group_test_dummy = area_group_test_dummy.astype("uint8")

#group_train과 같게 재배열
area_group_test_dummy = area_group_test_dummy[area_group_train_dummy.columns]

#비교
area_group_train_dummy.columns == area_group_test_dummy.columns

#원래행에 더미변수 조인 / 기존컬럼 삭제
group_train = group_train.join(area_group_train_dummy)
group_train = group_train.drop(["지역"], axis=1)

group_test = group_test.join(area_group_test_dummy)
group_test = group_test.drop(["지역"], axis=1)


##공급유형
supply_group_train_dummy = pd.get_dummies(group_train["공급유형"])
supply_group_test_dummy = pd.get_dummies(group_test["공급유형"])

supply_group_train_dummy.columns
supply_group_test_dummy.columns 

supply_group_test_dummy["공공분양"] = 0
supply_group_test_dummy["공공임대(5년)"] = 0
supply_group_test_dummy["장기전세"] = 0

supply_group_test_dummy = supply_group_test_dummy.astype("uint8")

supply_group_test_dummy.info()

supply_group_test_dummy = supply_group_test_dummy[supply_group_train_dummy.columns] 

supply_group_train_dummy.columns == supply_group_test_dummy.columns  

group_train = group_train.join(supply_group_train_dummy)
group_train = group_train.drop(["공급유형"], axis = 1)

group_test = group_test.join(supply_group_test_dummy)
group_test = group_test.drop(["공급유형"], axis = 1)

##신분
rank_group_train_dummy = pd.get_dummies(group_train["신분"])
rank_group_test_dummy = pd.get_dummies(group_test["신분"])

rank_group_train_dummy.columns
rank_group_test_dummy.columns 

rank_group_test_dummy["B"] = 0
rank_group_test_dummy["F"] = 0
rank_group_test_dummy["O"] = 0


rank_group_test_dummy = rank_group_test_dummy[rank_group_train_dummy.columns] 

rank_group_test_dummy = rank_group_test_dummy.astype("uint8")

rank_group_test_dummy.columns == rank_group_train_dummy.columns  

group_train = group_train.join(rank_group_train_dummy)
group_train = group_train.drop(["신분"], axis=1)

group_test = group_test.join(rank_group_test_dummy)
group_test = group_test.drop(["신분"], axis=1)


"""
#모델 xgboost
X_train = train.drop(["등록차량수"], axis = 1 )
y_train = train["등록차량수"]
X_test = test

import xgboost as xgb

xgb_model = xgb.XGBRegressor(booster='gbtree', 
                             n_estimators=100, 
                             learning_rate=0.08, 
                             gamma=0, 
                             subsample=0.75,
                             colsample_bytree=0.8,
                             colsample_bylevel=0.9,
                             nthread=4,
                             objective = "reg:linear",
                             max_depth=7,
                             random_state = 1572)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
"""


#명석 모델
from sklearn import preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV


X_train = group_train.drop(["등록차량수"], axis = 1 )
y_train = group_train["등록차량수"]
X_test = group_test

xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [3, 4, 5],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train,
         y_train)


print(xgb_grid.best_score_)
print(xgb_grid.best_params_)


#prediction
pred = xgb_grid.predict(X_test)


#병합
pred = pd.DataFrame(pred)
pred["code"] = group_test_index


for i in range(len(submission)):
   for j in range(len(pred)):
       if submission["code"][i] == pred["code"][j]:
           submission["num"][i] = pred[0][j]
           


           
#csv로 저장
submission.to_csv("submission5.csv", index=False)
