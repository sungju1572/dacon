import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv("test.csv")

train_corr = train.corr()

sns.heatmap(train_corr, cmap='viridis')
plt.show()


# 성능 변화 평가를 위해 data 나누기
X = train.iloc[:, 1:-1] 
Y = train.iloc[:, -1] 

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.5, shuffle=True, random_state=34)

from sklearn.metrics import f1_score

def print_score(predicted, y):
    accuracy = sum(predicted == y) / len(y)*100
    f1 = f1_score(y, predicted)*100
    print(f'accuracy: {accuracy:.2f}%') # 정확도 출력
    print(f'f1_score : {f1:.2f}%') # f1 score 출력
    return accuracy, f1


check_acc = []
check_f1 = []


#5단계 trian 데이터 준비
step_5_train_x = x_train.drop(["restecg","chol","fbs","trestbps"],axis=1)

#5단계 train
LR_step_5_model= LogisticRegression()
LR_step_5_model.fit(step_5_train_x,y_train)

#5단계 validation 데이터 준비
step_5_val_x = x_val.drop(["restecg","chol","fbs","trestbps"],axis=1)

#5단계 validation
LR_step_5_preds = LR_step_5_model.predict(step_5_val_x)
acc, f1 = print_score(LR_step_5_preds,y_val)

check_acc.append(acc)
check_f1.append(f1)

###
best_model_train_x = X.drop(["restecg","chol","fbs","trestbps"],axis=1)

best_model = LogisticRegression()
best_model.fit(best_model_train_x,Y)


#xgboost
from sklearn import preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV


#X_train = group_train.drop(["label"], axis = 1 ) #학습데이터
#y_train = group_train["label"] #정답라벨
#X_test = group_test #test데이터

xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [3, 4, 5],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500],
              "random_state" : [25]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(best_model_train_x,
         Y)


print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

