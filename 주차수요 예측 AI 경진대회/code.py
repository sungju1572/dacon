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

#임대보증금,임대료 제거
train = train.drop(["단지코드","임대보증금","임대료"], axis=1)

test = test.drop(["단지코드","임대보증금","임대료"], axis=1)

corr = train.corr()

train.isna().sum()
test.isna().sum()

#결측치처리
train = train.fillna(0)
test["신분"] = test["신분"].fillna("D")
test["지하철"] = test["지하철"].fillna(0)

####데이터 준비

###범주형 더미변수화
##임대건물구분
building_train_dummy = pd.get_dummies(train["임대건물구분"])
building_test_dummy = pd.get_dummies(test["임대건물구분"])

building_train_dummy.columns == building_test_dummy.columns 

train = train.join(building_train_dummy)
train = train.drop(["임대건물구분"], axis = 1)

test = test.join(building_test_dummy)
test = test.drop(["임대건물구분"], axis = 1)

##지역
area_train_dummy = pd.get_dummies(train["지역"])
area_test_dummy = pd.get_dummies(test["지역"])

#컬럼비교
area_train_dummy.columns  
area_test_dummy.columns 

#test에 없는 컬럼 채우기
area_test_dummy["서울특별시"] = 0
area_test_dummy = area_test_dummy.astype("uint8")

#train과 같게 재배열
area_test_dummy = area_test_dummy[area_train_dummy.columns]

#비교
area_train_dummy.columns == area_test_dummy.columns

#원래행에 더미변수 조인 / 기존컬럼 삭제
train = train.join(area_train_dummy)
train = train.drop(["지역"], axis=1)

test = test.join(area_test_dummy)
test = test.drop(["지역"], axis=1)


##공급유형
supply_train_dummy = pd.get_dummies(train["공급유형"])
supply_test_dummy = pd.get_dummies(test["공급유형"])

supply_train_dummy.columns
supply_test_dummy.columns 

supply_test_dummy["공공분양"] = 0
supply_test_dummy["공공임대(5년)"] = 0
supply_test_dummy["장기전세"] = 0

supply_test_dummy = supply_test_dummy.astype("uint8")

supply_test_dummy.info()

supply_test_dummy = supply_test_dummy[supply_train_dummy.columns] 

supply_train_dummy.columns == supply_test_dummy.columns  

train = train.join(supply_train_dummy)
train = train.drop(["공급유형"], axis = 1)

test = test.join(supply_test_dummy)
test = test.drop(["공급유형"], axis = 1)

##신분
rank_train_dummy = pd.get_dummies(train["신분"])
rank_test_dummy = pd.get_dummies(test["신분"])

rank_train_dummy.columns
rank_test_dummy.columns 

rank_test_dummy["B"] = 0
rank_test_dummy["F"] = 0
rank_test_dummy["O"] = 0


rank_test_dummy = rank_test_dummy[rank_train_dummy.columns] 

rank_test_dummy = rank_test_dummy.astype("uint8")

rank_test_dummy.columns == rank_train_dummy.columns  

train = train.join(rank_train_dummy)
train = train.drop(["신분"], axis=1)

test = test.join(rank_test_dummy)
test = test.drop(["신분"], axis=1)

#모델 xgboost
X_train = train.drop(["등록차량수"], axis = 1 )
y_train = train["등록차량수"]
X_test = test

import xgboost as xgb

xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)



#단지코드별로 평균내기
test1 = pd.read_csv('test.csv')

y_pred = pd.DataFrame(y_pred)

y_pred["code"] = test1["단지코드"]

grouping = y_pred.groupby(y_pred.code).mean()

grouping.columns = ["num"]

grouping["num"]
grouping["code"] = grouping.index

pd.merge(submission, grouping, how="left", left_on = "num", right_on = "num")

#submission 에 합치기
for i in range(len(submission)):
   for j in range(len(grouping)):
       if submission["code"][i] == grouping["code"][j]:
           submission["num"][i] = grouping["num"][j]
           
           
           
#csv로 저장
submission.to_csv("submission1.csv", index=False)
