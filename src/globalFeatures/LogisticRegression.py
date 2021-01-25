import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

y_train = train_data.iloc[:,0]
y_test = test_data.iloc[:,0]
X_train = train_data.iloc[:,1:]
X_test = test_data.iloc[:,1:]

print('[INFO] y_train dim:', y_train.shape)
print('[INFO] y_test dim:', y_test.shape)
print('[INFO] X_train:', X_train.shape)
print('[INFO] X_train dim:', X_test.shape)

### basic model ###
start_time = time.time()
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))
print(round(time.time() - start_time, 2))

### model with scaler ###
start_time = time.time()
steps = [('scaler', StandardScaler()),
         ('classifier', LogisticRegression(max_iter=10000))]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

print(pipeline.score(X_test, y_test))
print(round(time.time() - start_time, 2))

### model with scaler and hyperparameter tuning ###
start_time = time.time()
steps = [('scaler', StandardScaler()),
         ('classifier', LogisticRegression(max_iter=10000))]

params_space = {
    'classifier__solver': ['lbfgs', 'liblinear'],
    'classifier__C': [0.001, 0.01, 0.1, 1.0, 10],
    'classifier__penalty': ['l2']
}

pipeline = Pipeline(steps)

gs_logit = GridSearchCV(estimator=pipeline,
                        param_grid=params_space,
                        scoring='neg_root_mean_squared_error',
                        cv=10,
                        verbose=0)

gs_logit.fit(X_train, y_train)

for i in range(len(gs_logit.cv_results_['params'])):
    print(gs_logit.cv_results_['params'][i], 'test RMSE:', gs_logit.cv_results_['mean_test_score'][i])


print(gs_logit.best_estimator_.score(X_test, y_test))
print(round(time.time() - start_time, 2))