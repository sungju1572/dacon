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

#train = train.drop(["id"], axis=1)
#test = test.drop(["id"], axis=1)


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

X.info()


#시각화
plt.style.use("ggplot")

# 히스토그램 을 사용해서 데이터의 분포를 살펴봅니다.
plt.figure(figsize=(25,20))
plt.suptitle("Data Histogram", fontsize=40)

# id는 제외하고 시각화합니다.
cols = train.columns[1:]
for i in range(len(cols)):
    plt.subplot(5,3,i+1)
    plt.title(cols[i], fontsize=20)
    if len(train[cols[i]].unique()) > 20:
        plt.hist(train[cols[i]], bins=20, color='b', alpha=0.7)
    else:
        temp = train[cols[i]].value_counts()
        plt.bar(temp.keys(), temp.values, width=0.5, alpha=0.7)
        plt.xticks(temp.keys())
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()






###train
best_model_train_x = X.drop(["restecg","fbs"],axis=1)

a = best_model_train_x.reset_index()
best_model_train_x = a.drop(["index"], axis=1)

best_model_train_x["age"].describe

#age 범주형변환
for i in range(0,len(best_model_train_x)):
    if best_model_train_x["age"][i]>=10 and best_model_train_x["age"][i]<20:
        best_model_train_x["age"][i] = 1
    elif best_model_train_x["age"][i]>=20 and best_model_train_x["age"][i]<30:
        best_model_train_x["age"][i] = 2
    elif best_model_train_x["age"][i]>=30 and best_model_train_x["age"][i]<40:
        best_model_train_x["age"][i] = 3
    elif best_model_train_x["age"][i]>=40 and best_model_train_x["age"][i]<50:
        best_model_train_x["age"][i] = 4
    elif best_model_train_x["age"][i]>=50 and best_model_train_x["age"][i]<60:
        best_model_train_x["age"][i] = 5
    elif best_model_train_x["age"][i]>=60 and best_model_train_x["age"][i]<70:
        best_model_train_x["age"][i] = 6
    elif best_model_train_x["age"][i]>=70 and best_model_train_x["age"][i]<80:
        best_model_train_x["age"][i] = 7
    print(i)
        
#oldpeak 범주형 변환
for i in range(0,len(best_model_train_x)):
    if best_model_train_x["oldpeak"][i]>0 and best_model_train_x["oldpeak"][i]<=0.5:
        best_model_train_x["oldpeak"][i] = 1
    elif best_model_train_x["oldpeak"][i]> 0.5 and best_model_train_x["oldpeak"][i]<=1.5:
        best_model_train_x["oldpeak"][i] = 2
    elif best_model_train_x["oldpeak"][i]> 1.5 and best_model_train_x["oldpeak"][i]<=2.5:
        best_model_train_x["oldpeak"][i] = 3
    elif best_model_train_x["oldpeak"][i]> 2.5 and best_model_train_x["oldpeak"][i]<=3.5:
        best_model_train_x["oldpeak"][i] = 4
    elif best_model_train_x["oldpeak"][i]> 3.5 and best_model_train_x["oldpeak"][i]<= 4.5:
        best_model_train_x["oldpeak"][i] = 5
    elif best_model_train_x["oldpeak"][i]> 4.5 and best_model_train_x["oldpeak"][i]<= 5.5:
        best_model_train_x["oldpeak"][i] = 6




#best_model_train_x = best_model_train_x[best_model_train_x["chol"] != 564]


#test
best_model_test = test.drop(["id","restecg","fbs"],axis=1)


#age 범주형변환
for i in range(0,len(best_model_test)):
    if best_model_test["age"][i]>=10 and best_model_test["age"][i]<20:
        best_model_test["age"][i] = 1
    elif best_model_test["age"][i]>=20 and best_model_test["age"][i]<30:
        best_model_test["age"][i] = 2
    elif best_model_test["age"][i]>=30 and best_model_test["age"][i]<40:
        best_model_test["age"][i] = 3
    elif best_model_test["age"][i]>=40 and best_model_test["age"][i]<50:
        best_model_test["age"][i] = 4
    elif best_model_test["age"][i]>=50 and best_model_test["age"][i]<60:
        best_model_test["age"][i] = 5
    elif best_model_test["age"][i]>=60 and best_model_test["age"][i]<70:
        best_model_test["age"][i] = 6
    elif best_model_test["age"][i]>=70 and best_model_test["age"][i]<80:
        best_model_test["age"][i] = 7
        
#oldpeak 범주형 변환
for i in range(0,len(best_model_test)):
    if best_model_test["oldpeak"][i]>0 and best_model_test["oldpeak"][i]<=0.5:
        best_model_test["oldpeak"][i] = 1
    elif best_model_test["oldpeak"][i]> 0.5 and best_model_test["oldpeak"][i]<=1.5:
        best_model_test["oldpeak"][i] = 2
    elif best_model_test["oldpeak"][i]> 1.5 and best_model_test["oldpeak"][i]<=2.5:
        best_model_test["oldpeak"][i] = 3
    elif best_model_test["oldpeak"][i]> 2.5 and best_model_test["oldpeak"][i]<=3.5:
        best_model_test["oldpeak"][i] = 4
    elif best_model_test["oldpeak"][i]> 3.5 and best_model_test["oldpeak"][i]<= 4.5:
        best_model_test["oldpeak"][i] = 5
    elif best_model_test["oldpeak"][i]> 4.5 and best_model_test["oldpeak"][i]<= 5.5:
        best_model_test["oldpeak"][i] = 6



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



##예측


pred = xgb_grid.predict(best_model_test)

sub= list(map(lambda x : 1 if x>0.5 else 0, pred))

#sub
submission = pd.read_csv('sample_submission.csv')
submission['target'] = sub

submission.to_csv('submission.csv', index=False)


########
all_data = pd.concat([train, test], sort=False)
print(all_data.shape)
all_data.head(3)


for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
    print(all_data[col].value_counts())
    print('==='*10)
    
# cardinality가 low한 값을 nan으로 변경
all_data.loc[all_data['restecg'] == 2, 'restecg'] = np.nan
all_data.loc[all_data['ca'] == 4, 'ca'] = np.nan
all_data.loc[all_data['thal'] == 0, 'thal'] = np.nan


#!pip install pycaret
from pycaret.classification import * 

ignore_features=['id']
categorical_features=['sex', 'cp', 'restecg', 'exang', 'slope', 'ca', 'thal']
numeric_features=['age', 'trestbps', 'chol', 'fbs', 'thalach', 'oldpeak', ]

# 전처리 pipeline setup
clf = setup(data=train, 
            target='target', 
            ignore_features=ignore_features,           # 분석/학습에 고려하지 않을 feature(컬럼) 제거
            categorical_features=categorical_features, # 범주형 컬럼 지정
            numeric_features=numeric_features,         # 수치형 컬럼 지정
            normalize=True,                            # 정규화 적용
            normalize_method='zscore',                 # 정규화 방식 지정
            imputation_type='iterative',               # 결측치를 lightgbm으로 예측하여 채움
            iterative_imputation_iters=10,             # imputation iteration 지정
            categorical_iterative_imputer='lightgbm',
            # bin_numeric_features=['age', 'trestbps', 'chol','thalach', 'oldpeak'], # numeric 컬럼에 대하여 binning
            polynomial_features=True,
            session_id=123, 
            silent=True,
            ) 

best_models = compare_models(sort='f1', n_select=5, fold=5)



X = train.iloc[:, 1:-1] 
Y = train.iloc[:, -1] 

  
x_train, x_val, y_train, y_val = model_selection.train_test_split(X, Y)

exp = setup(data = train, 
            target="target", 
            session_id = 1234,
            normalize=True)


best = compare_models(sort="f1", n_select=5,fold=5)

blended_models = blend_models(best, fold=5)

finalized_models = finalize_model(blended_models)

predictions = predict_model(data=test, estimator=finalized_models)
predictions


submission = pd.read_csv('sample_submission.csv')



submission['target'] = predictions['Label']
submission.to_csv('output.csv', index=False)


##

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
np.random.seed(0) #for reproducibility
#
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


X = train.iloc[:, 1:-1] 
Y = train.iloc[:, -1] 

  
x_train, x_val, y_train, y_val = model_selection.train_test_split(X, Y)

  
def scatter1():
    import matplotlib.pylab as plt
    import matplotlib as mpl
    mpl.rc('font', family='Malgun Gothic') #한글 폰트 설정
    #
    plt.scatter(x_train[:,0][y_train==0], x_train[:,1][y_train==0], label='0 (train)', color='red')
    plt.scatter(x_train[:,0][y_train==1], x_train[:,1][y_train==1], label='1 (train)', color='green')

    #
    plt.scatter(x_val[:,0][y_val==0], x_test[:,1][y_val==0], label='0 (test)', color='red', marker='x')
    plt.scatter(x_val[:,0][y_val==1], x_test[:,1][y_val==1], label='1 (test)', color='green', marker='x')

    #
    plt.title('분포도')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend() #범례
    plt.show()
scatter1()
