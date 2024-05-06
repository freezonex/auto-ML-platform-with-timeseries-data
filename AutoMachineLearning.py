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
            print("Testing parameters:", params)
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
            losses.extend(loss)
        return sum(losses) / len(losses)

# Base AutoML Class: Abstract base class for automated machine learning workflows
class AutoMLBase(ABC):
    def __init__(self, data, label):
        """Initialize with data and labels.
        Use case: Forms the base for any AutoML system that automates the process of applying machine learning models to real-world tasks."""
        self.data = data
        self.label = label
        self.models = {}
        self.tuners = {}

    def add_model(self, name, model):
        """Add a model to the AutoML system.
        Use case: Allows dynamic addition of models to the system for experimentation or deployment."""
        self.models[name] = model

    def add_tuner(self, name, tuner):
        """Add a tuner to the AutoML system.
        Use case: Supports extending the system with new tuning strategies without modifying the core architecture."""
        self.tuners[name] = tuner

    @abstractmethod
    def run_experiments(self):
        """Run experiments using the added models and tuners.
        Use case: Used to conduct extensive testing and validation of models across different datasets."""
        pass

# Concrete Implementation for Time Series AutoML
class TimeSeriesAutoML(AutoMLBase):
    def run_experiments(self):
        """Run experiments on time series data using the specified models and tuners.
        Use case: Ideal for applications in financial markets, weather forecasting, and demand forecasting where time series analysis is crucial."""
        train_data = []
        train_label = []
        for key in self.data.keys():
            train_data.extend(np.array(self.data[key]))
            train_label.extend(np.array(self.label[key]))
        train_data = np.array(train_data)
        train_label = np.array(train_label)
        for name, model in self.models.items():
            param = param_grid[name]
            for name_tuner, tuner in self.tuners.items():
                model_class, score = tuner.optimize(model, param, train_data, train_label)
                model_class.save(f'{name}_{name_tuner}_{score}_best.pth')

    def evaluate(self, predictions, actual):
        """Evaluate predictions against actual data and return a metric.
        Use case: Provides a method to assess the effectiveness of models in making accurate predictions."""
        return np.mean(predictions == actual)  # Example metric



