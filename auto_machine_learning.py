from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, f1_score
import joblib
import os
import numpy as np
from model.models import NeuralNetRegressor
def AUTOML(x,y,mode='regression'):
    model_path = 'best_model/model.pkl'
    #we onlt use random forest, svm and xgboost here for demo
    # #data pre-processing-normalization
    x_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x)


    models_params = get_model_parameters(mode=mode)

    scores = {}
    if os.path.exists(model_path):
        print("Loading the existing model...")
        best_model = joblib.load(model_path)

    #
    else:
        for model_name, mp in models_params.items():
            clf = GridSearchCV(mp['model'], mp['params'], cv=5,
                               scoring='neg_mean_squared_error' if mode == 'regression' else 'f1')
            clf.fit(x_train_scaled, y)
            scores[model_name] = clf.best_score_
            models_params[model_name]['model'] = clf.best_estimator_

        best_model_name = max(scores, key=scores.get)
        best_model = models_params[best_model_name]['model']

    # best_model = xgb.XGBRegressor(
    #     objective = 'reg:squarederror',
    #     max_depth = 10,
    #     n_estimators = 50,
    #     learning_rate = 0.1
    # )
    # # Retrain the best model on the entire dataset
    best_model.fit(x_train_scaled, y)

    # Optionally: evaluate the model on the test set or return it
    if mode == 'regression':
        test_score = mean_squared_error(y, best_model.predict(x_train_scaled),squared=False)
    else:
        test_score = f1_score(y, best_model.predict(x_train_scaled))

    print(f"Best Model Train Score: {test_score}")
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    return best_model,x_scaler

def get_model_parameters(mode,is_time_series=False):
    # Set up the model and parameters
    if not is_time_series:
        models_params = {
            # 'random_forest': {
            #     'model': RandomForestClassifier() if mode == 'classification' else RandomForestRegressor(),
            #     'params': {
            #         'n_estimators': [10, 50, 100],
            #         'max_depth': [None, 10, 20, 30]
            #     }
            # },
            # 'svm': {
            #     'model': SVC() if mode == 'classification' else SVR(),
            #     'params': {
            #         'C': [0.1, 1, 10],
            #         'kernel': ['rbf', 'linear']
            #     }
            # },
            'xgboost': {
                'model': xgb.XGBClassifier() if mode == 'classification' else xgb.XGBRegressor(),
                'params': {
                    'n_estimators': [10,50,100],
                    'max_depth': [ 3,5,10],
                    'learning_rate': [ 0.1,0.01]
                }
            }
        }
    else:
        models_params = {
            'lstm': {
                'model': NeuralNetRegressor,
                'params' : {
                    'module__hidden_dim': [ 30, 50],
                    'lr': [0.01, 0.001],
                    'batch_size': [16, 32]
                }
            }

        }
    return models_params

def AUTOML_for_time_series(dataframe,label,group_by=None):
    scaler = MinMaxScaler()
    scaled_dataframe = scaler.fit_transform(dataframe['storage'].values.reshape(-1, 1))

    look_backs = [2,3,4]
    for look_back in look_backs:
        train_datasets, train_labels, test_datasets, test_labels = create_time_series_data(dataframe,label,look_back,group_by=group_by)


def create_time_series_data(dataframe,label,look_back,group_by=None):
    train_datasets = []
    test_datasets = []
    train_labels = []
    test_labels = []

    if group_by:
        for name, group in dataframe.groupby(group_by):
            split_index = int(len(group) * 0.8)
            train_group = group.iloc[:split_index]
            test_group = group.iloc[split_index - look_back:]

            for start_index in range(len(train_group) - look_back):
                subset = train_group.iloc[start_index:start_index + look_back]
                truth = train_group.iloc[start_index + look_back][label]
                train_datasets.append(subset)
                train_labels.append(truth)

            for start_index in range(len(test_group) - look_back):
                subset = test_group.iloc[start_index:start_index + look_back]
                truth = test_group.iloc[start_index + look_back][label]
                test_datasets.append(subset)
                test_labels.append(truth)
    else:
        split_index = int(len(dataframe)*0.8)
        train_group = dataframe.iloc[:split_index]
        test_group = dataframe.iloc[split_index - look_back:]
        train_datasets.append(train_group)
        test_datasets.append(test_group)
    return train_datasets, train_labels, test_datasets, test_labels