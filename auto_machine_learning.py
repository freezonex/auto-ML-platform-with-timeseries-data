from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, f1_score
import joblib
import os
import numpy as np
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

def get_model_parameters(mode):
    # Set up the model and parameters
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
    return models_params