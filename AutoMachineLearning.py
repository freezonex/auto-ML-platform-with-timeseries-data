import os
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error,f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from utlis.preprocessing import TimeSeriesPreprocessor
from abc import ABC, abstractmethod


# Model Interface
class Model(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

# Tuner Interface
class Tuner(ABC):
    @abstractmethod
    def optimize(self, model, X, y):
        pass

# Base AutoML Class
class AutoMLBase(ABC):
    def __init__(self, data: pd.DataFrame, target_column: str):
        self.data = data
        self.target = target_column
        self.models = {}
        self.tuners = {}

    def add_model(self, name, model):
        self.models[name] = model

    def add_tuner(self, name, tuner):
        self.tuners[name] = tuner

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def run_experiments(self):
        pass

# Concrete Implementation for Time Series AutoML
class TimeSeriesAutoML(AutoMLBase):
    def preprocess_data(self):
        # Specific preprocessing for time series
        print("Preprocessing time series data")

    def run_experiments(self):
        train_data, test_data = train_test_split(self.data)
        for name, model in self.models.items():
            tuner = self.tuners.get(name)
            if tuner:
                best_params = tuner.optimize(model, train_data, train_data[self.target])
                model.set_params(**best_params)
            model.train(train_data.drop(columns=[self.target]), train_data[self.target])
            predictions = model.predict(test_data.drop(columns=[self.target]))
            self.evaluate(predictions, test_data[self.target])

    def evaluate(self, predictions, actual):
        # Implement evaluation logic
        return np.mean(predictions == actual)  # Example metric

# Usage
data = pd.DataFrame()  # Your actual data
ts_automl = TimeSeriesAutoML(data, 'target_column')
ts_automl.add_model('ARIMA', ArimaModel())
ts_automl.add_tuner('ARIMA', GridSearchTuner(params))
ts_automl.run_experiments()
