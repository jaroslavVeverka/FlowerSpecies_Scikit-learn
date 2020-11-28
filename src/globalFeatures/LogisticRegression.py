import time
import numpy as np
import pandas as pd

# import custom functions for global features extraction
from src.globalFeatures.feature_extraction import extract_global_features, prepare_images

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# path to data
data_dir = 'C:/Users/jarda/IdeaProjects/FlowerSpecies_Scikit-learn/src/dataset/train/'

# get labeled images
labeled_images = prepare_images(data_dir)
print('[STATUS] data size: ', np.array(labeled_images).shape)

images = [image[1] for image in labeled_images]
# get Y
labels = [image[0] for image in labeled_images]

# get extracted features of images: X
extracted_features = extract_global_features(images)

X_train, X_test, y_train, y_test = train_test_split(np.array(extracted_features),
                                                    np.array(labels),
                                                    test_size=0.2, stratify=np.array(labels), random_state=42)

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