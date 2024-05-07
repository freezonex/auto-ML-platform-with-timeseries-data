import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class TimeSeriesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, config,feature_scaler=None, label_scaler=None, look_back=1):
        self.feature_scaler = feature_scaler if feature_scaler else MinMaxScaler()
        self.label_scaler = label_scaler if label_scaler else MinMaxScaler()
        self.look_back = look_back
        self.group_col = config.group_by
        self.label_col = config.label
        self.task = config.task
        self.timestamp = config.timestamp_column

    def fit(self, X, y=None):
        features = X.drop(columns=[self.label_col] if not self.group_col else [self.label_col,self.group_col] , errors='ignore')
        self.feature_scaler.fit(features)
        if self.label_col in X.columns:
            self.label_scaler.fit(X[[self.label_col]])
        return self

    def transform(self, X):
        features_to_scale = X.drop(columns=[self.label_col] if not self.group_col else [self.label_col,self.group_col] , errors='ignore')
        features = features_to_scale.columns
        scaled_features = pd.DataFrame(self.feature_scaler.transform(features_to_scale),
                                       columns=features_to_scale.columns,
                                       index=X.index)
        # If the label column exists and needs to be included in the output
        if self.label_col in X.columns:
            scaled_features[self.label_col] = self.label_scaler.transform(X[[self.label_col]])

        if self.group_col and self.group_col in X.columns:
            scaled_features[self.group_col] = X[self.group_col]

        # Now group by if necessary and create lagged features
        time_series_data = self._create_time_series_data(scaled_features,features)
        return time_series_data
    def _create_time_series_data(self, dataframe, features):
        data_dict = {}
        label_dict = {}
        if self.group_col:
            for name,group in dataframe.groupby(self.group_col):
                data = []
                label = []
                for start_index in range(len(group) - self.look_back):
                    subset = group.iloc[start_index:start_index + self.look_back][features]
                    sublabel = group.iloc[start_index + self.look_back-1][self.label_col]
                    data.append(subset)
                    label.append(sublabel)
                data_dict[name] = data
                label_dict[name] = label
        else:
            data = []
            label = []
            for start_index in range(len(dataframe) - self.look_back):
                subset = dataframe.iloc[start_index:start_index + self.look_back][features]
                sublabel = dataframe.iloc[start_index + self.look_back - 1][self.label_col]
                data.append(subset)
                label.append(sublabel)
            data_dict['no group'] = data
            label_dict['no group'] = label

        return data_dict,label_dict

    def inverse_transform_labels(self, y_scaled):
        """Reverse the scaling of the label data."""
        return self.label_scaler.inverse_transform(y_scaled.reshape(-1, 1))

    def save_scalers(self, feature_scaler_path, label_scaler_path=None):
        joblib.dump(self.feature_scaler, feature_scaler_path)
        if label_scaler_path and self.label_scaler:
            joblib.dump(self.label_scaler, label_scaler_path)

    def load_scalers(self, feature_scaler_path, label_scaler_path=None):
        self.feature_scaler = joblib.load(feature_scaler_path)
        if label_scaler_path:
            self.label_scaler = joblib.load(label_scaler_path)
