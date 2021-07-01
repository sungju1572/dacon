import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
age = pd.read_csv("age_gender_info.csv")

plt.plot(train["단지코드"], train["등록차량수"])

##데이터확인

train.info()

#임대보증금,임대료 타입 변경

train = train.astype({"임대보증금" : "float", "임대료":"float"})

train.columns

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

#


corr = train.corr()

##데이터 준비

#임대보증금,임대료 NA값 다 지우고 float형으로 전환후 평균값 구해서 채워넣기
rental_d = train[["임대보증금","임대료","등록차량수"]] 
rental_d.isna().sum()
rental_d  = rental_d.dropna()
rental_d
# - 들가는 행 찾기
idx = rental_d[rental_d["임대보증금"].str.contains('-')].index

rental_d = rental_d.drop(idx)

rental_d= rental_d.astype({"임대보증금" : "float", "임대료":"float"})
a = rental_d.corr()

rental_d.info()
rental_d[rental_d["임대료"=='-']]
