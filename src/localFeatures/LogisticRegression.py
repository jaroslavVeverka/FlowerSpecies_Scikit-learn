# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:21:47 2021

@author: jveverka
"""

import time
import numpy as np
import pandas as pd

# import custom functions for global features extraction
from src.localFeatures.feature_extraction import  prepare_images, extract_local_features, fit_transform_bovw, transform_bovw
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

train_trans_bovw = pd.read_csv('train_trans_bovw.csv')
test_trans_bovw = pd.read_csv('test_trans_bovw.csv')

y_train = train_trans_bovw.iloc[:,0]
y_test = test_trans_bovw.iloc[:,0]
X_train_trans_bovw = train_trans_bovw.iloc[:,1:]
X_test_trans_bovw = test_trans_bovw.iloc[:,1:]

print('[INFO] y_train dim:', y_train.shape)
print('[INFO] y_test dim:', y_test.shape)
print('[INFO] X_train_trans_bovw:', X_train_trans_bovw.shape)
print('[INFO] X_test_trans_bovw dim:', X_test_trans_bovw.shape)


### basic model ###
print('[INFO] basic model LR')
start_time = time.time()

# specify model and fit
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train_trans_bovw, y_train)

# use model on test data
y_pred = lr.predict(X_test_trans_bovw)

# evaluate test predictions
print('[STATUS] Acc. score on test data: ', lr.score(X_test_trans_bovw, y_test))
conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)
print(classification_report(y_test, y_pred))

print(round(time.time() - start_time, 2))

### model with scaler ###
print('[INFO] model LR with scaler')
start_time = time.time()
steps = [('scaler', StandardScaler()),
         ('classifier', LogisticRegression(max_iter=10000))]

pipeline = Pipeline(steps)

# fit pipeline
pipeline.fit(X_train_trans_bovw, y_train)

# use pipeline on test data
y_pred = pipeline.predict(X_test_trans_bovw)

# evaluate test predictions
print('[STATUS] Acc. score on test data: ', pipeline.score(X_test_trans_bovw, y_test))
conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)
print(classification_report(y_test, y_pred))

print(round(time.time() - start_time, 2))

### model with scaler and hyperparameter tuning ###
print('[INFO] model LR with scaler and hyperparameter tuning')
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

gs_logit.fit(X_train_trans_bovw, y_train)

for i in range(len(gs_logit.cv_results_['params'])):
    print(gs_logit.cv_results_['params'][i], 'test RMSE:', gs_logit.cv_results_['mean_test_score'][i])

y_pred = gs_logit.best_estimator_.predict(X_test_trans_bovw)

print('[STATUS] Acc. score on test data: ', 
      gs_logit.best_estimator_.score(X_test_trans_bovw, y_test))
conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)
print(classification_report(y_test, y_pred))

print(round(time.time() - start_time, 2))

### model with scaler and hyperparameter tuning + PCA ###
print('[INFO] model LR with scaler and hyperparameter tuning + PCA')
start_time = time.time()
from sklearn.decomposition import PCA

steps = [('scaler', StandardScaler()),
         ('pca', PCA()),
         ('classifier', LogisticRegression(max_iter=10000))]

params_space = {
    'pca__n_components': [5, 10, 15, 20, 30, 40, 50, 60, 120],
    'classifier__solver': ['lbfgs', 'liblinear'],
    'classifier__C': [0.001, 0.01, 0.1, 1.0, 10],
    'classifier__penalty': ['l2']
}

pipeline = Pipeline(steps)

gs_logit_pca = GridSearchCV(estimator=pipeline,
                        param_grid=params_space,
                        scoring='neg_root_mean_squared_error',
                        cv=10,
                        verbose=0)

gs_logit_pca.fit(X_train_trans_bovw, y_train)

for i in range(len(gs_logit_pca.cv_results_['params'])):
    print(gs_logit_pca.cv_results_['params'][i], 'test RMSE:', gs_logit_pca.cv_results_['mean_test_score'][i])

y_pred = gs_logit_pca.best_estimator_.predict(X_test_trans_bovw)

print('[STATUS] Acc. score on test data: ', 
      gs_logit_pca.best_estimator_.score(X_test_trans_bovw, y_test))
conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)
print(classification_report(y_test, y_pred))

print(round(time.time() - start_time, 2))

