import os
import joblib
import numpy as np
import itertools
import pandas as pd
import math
from collections import defaultdict
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error,f1_score
from sklearn.base import BaseEstimator
from utlis.preprocessing import TimeSeriesPreprocessor
from sklearn.model_selection import KFold
from model.models import param_grid
# Model Interface



# Tuner Interface: Abstract base class for tuning models
class Tuner(ABC):
    @abstractmethod
    def optimize(self, model_class, param_grid, X, y):
        """Optimize model parameters over the defined parameter grid.
        Use case: Can be used in ML workflows to find the best model parameters for prediction accuracy."""
        pass

    @abstractmethod
    def cross_validation(self, model_class, params, X, y):
        """Perform cross-validation and return the average loss.
        Use case: Useful in validating the model's performance to avoid overfitting on various datasets."""
        pass

# GridSearchTuner: Implements parameter tuning by grid search method
class GridSearchTuner(Tuner):
    def __init__(self, num_folds=5):
        """Initialize the tuner with the number of folds for cross-validation.
        Use case: Typically used in scenarios where robust model evaluation is needed across multiple subsets of data."""
        self.num_folds = num_folds
        self.best_params = None
        self.best_score = float('inf')

    def optimize(self, model_class, param_grid, X, y):
        """Perform grid search optimization over a parameter grid for the given model.
        Use case: Ideal for optimizing models in settings where multiple hyperparameters need systematic evaluation."""
        keys, values = zip(*param_grid.items())
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            avg_loss = self.cross_validation(model_class, params, X, y)
            if avg_loss < self.best_score:
                self.best_score = avg_loss
                self.best_params = params
            print(f"Testing parameters:{params} loss:{avg_loss}")
        print("Best parameters:", self.best_params)
        print("Best score:", self.best_score)
        model_class.set_params(**self.best_params)
        model_class.train(X, y)
        return model_class, self.best_score

    def cross_validation(self, model_class, params, X, y):
        """Perform k-fold cross-validation and return the average loss across all folds.
        Use case: Essential for assessing the generalization capability of models in academic research or industrial applications."""
        kfold = KFold(n_splits=self.num_folds, shuffle=True)
        losses = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model_class.set_params(**params)
            model_class.train(X_train, y_train)
            loss = model_class.evaluate(X_test, y_test)
            losses.append(loss)
        return sum(losses) / len(losses)
    def reset(self):
        self.best_params = None
        self.best_score = float('inf')

# Base AutoML Class: Abstract base class for automated machine learning workflows
class AutoMLBase(ABC):
    def __init__(self):
        """Initialize with data and labels.
        Use case: Forms the base for any AutoML system that automates the process of applying machine learning models to real-world tasks."""
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.models = {}
        self.tuners = {}
        self.best_model = {}
        self.best_score = {}
        self.train_result = {}
        self.test_result = {}
    def add_model(self, name, model):
        """Add a model to the AutoML system.
        Use case: Allows dynamic addition of models to the system for experimentation or deployment."""
        self.models[name] = model

    def add_tuner(self, name, tuner):
        """Add a tuner to the AutoML system.
        Use case: Supports extending the system with new tuning strategies without modifying the core architecture."""
        self.tuners[name] = tuner

    @abstractmethod
    def run_experiments(self,train_data,train_label):
        """Run experiments using the added models and tuners.
        Use case: Used to conduct extensive testing and validation of models across different datasets."""
        pass

    @abstractmethod
    def evaluate(self,test_data,test_label):
        """Run experiments using the added models and tuners.
        Use case: Used to conduct extensive testing and validation of models across different datasets."""
        pass

# Concrete Implementation for Time Series AutoML
class TimeSeriesAutoML(AutoMLBase):
    def run_experiments(self,train_data,train_label):
        """
        Run experiments on time series data using the specified models and tuners.
        Save the best model path for each kind of model
        Use case: Ideal for applications in financial markets, weather forecasting, and demand forecasting
        where time series analysis is crucial."""
        data = []
        label = []
        for key in train_data.keys():
            data.extend(np.array(train_data[key]))
            label.extend(np.array(train_label[key]))
        train_data = np.array(data)
        train_label = np.array(label)
        self.train_data = train_data
        self.train_label = train_label
        for name, model in self.models.items():
            best_score = float('inf')
            param = param_grid[name]
            for name_tuner, tuner in self.tuners.items():
                model_class, score = tuner.optimize(model, param, train_data, train_label)
                tuner.reset()
                model_class.save(f'{name}_{name_tuner}_{score}_best.pth')
                if score < best_score:
                    best_score = score
                    self.best_score[name] = score
                    self.best_model[name] = f'{name}_{name_tuner}_{score}_best.pth'
                train_prediction = model.predict(self.train_data)
                self.train_result[name] = {
                    'train_loss': self.best_score[name],
                    'train_prediction': train_prediction
                }
    def evaluate(self, test_data, test_label):
        """evaluation on train data and test data to give a result of the model, can be the best model or the
        best model in each kind of model.
        Use case: Provides a method to assess the effectiveness of models in making accurate predictions."""
        data = []
        label = []
        for key in test_data.keys():
            data.extend(np.array(test_data[key]))
            label.extend(np.array(test_label[key]))
        test_data = np.array(data)
        test_label = np.array(label)
        self.test_data = test_data
        self.test_label = test_label
        for name, model in self.models.items():
            model.load(self.best_model[name])
            test_loss = model.evaluate(self.test_data,self.test_label)
            test_prediction = model.predict(self.test_data)

            self.test_result[name] = {
                'test_loss': test_loss,
                'test_prediction': test_prediction
            }






