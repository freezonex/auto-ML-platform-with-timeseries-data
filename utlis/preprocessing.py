import pandas as pd
import joblib
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class TimeSeriesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, config,feature_scaler=None, label_scaler=None, look_back=1,predict_time_stamp=1):
        self.look_back = look_back
        self.group_col = config.group_by
        self.label_col = config.label
        self.task = config.task
        self.timestamp = config.timestamp_column
        self.predict_time_stamp = predict_time_stamp
        self.include_label = config.include_label
        self.num_features = None
        self.features = None

        os.makedirs(f'static/scaler/{self.task}', exist_ok=True)
        if os.path.exists(f'static/scaler/{self.task}/feature.pkl'):
            if os.path.exists(f'static/scaler/{self.task}/label.pkl'):
                self.load_scalers(f'static/scaler/{self.task}/feature.pkl',
                                                f'static/scaler/{self.task}/label.pkl')
            else:
                self.load_scalers(f'static/scaler/{self.task}/feature.pkl',
                                  None)
            print(f'loaded scalers for {self.task}')
            self.loaded_scaler = True
        else:
            self.feature_scaler = feature_scaler if feature_scaler else StandardScaler()
            self.label_scaler = label_scaler if label_scaler else StandardScaler()
            self.loaded_scaler = False
    def fit(self, X, y=None):
        features_to_scale = X.drop(columns=[self.label_col] if not self.group_col else [self.label_col,self.group_col] , errors='ignore')
        self.features = features_to_scale.columns

        if not self.loaded_scaler:
            self.feature_scaler.fit(features_to_scale)
            if self.label_col in X.columns:
                self.label_scaler.fit(X[[self.label_col]])
                self.save_scalers(f'static/scaler/{self.task}/feature.pkl',
                                  f'static/scaler/{self.task}/label.pkl')
            else:
                self.save_scalers(f'static/scaler/{self.task}/feature.pkl',)


    def transform(self, X):
        features_to_scale = X.drop(columns=[self.label_col] if not self.group_col else [self.label_col,self.group_col] , errors='ignore')
        self.features = features_to_scale.columns
        scaled_features = pd.DataFrame(self.feature_scaler.transform(features_to_scale),
                                       columns=features_to_scale.columns,
                                       index=X.index)
        # If the label column exists and needs to be included in the output
        if self.label_col in X.columns:
            scaled_features[self.label_col] = self.label_scaler.transform(X[[self.label_col]])

        if self.group_col and self.group_col in X.columns:
            scaled_features[self.group_col] = X[self.group_col]

        # Now group by if necessary and create lagged features
        time_series_data = self._create_time_series_data(scaled_features)
        return time_series_data
    def _create_time_series_data(self, dataframe):
        data_dict = {}
        label_dict = {}
        has_label = self.label_col in dataframe.columns
        if self.include_label:
            self.features = list(self.features) + [self.label_col]
        if self.group_col:
            for name, group in dataframe.groupby(self.group_col):
                data = []
                label = []
                for start_index in range(len(group) - self.look_back-self.predict_time_stamp+1):
                    if has_label:
                        subset = group.iloc[start_index:start_index + self.look_back][self.features]
                        sublabel = group[self.label_col].iloc[
                                   start_index + self.look_back:start_index + self.look_back + self.predict_time_stamp
                                   ].reset_index(drop=True).values
                        label.append(sublabel)
                    else:
                        subset = group.iloc[start_index:start_index + self.look_back][self.features]
                    data.append(subset)

                data_dict[name] = data
                if has_label:
                    label_dict[name] = label
        else:
            data = []
            label = []
            for start_index in range(len(dataframe) - self.look_back-self.predict_time_stamp+1):
                if has_label:
                    subset = dataframe.iloc[start_index:start_index + self.look_back][self.features]
                    sublabel = dataframe[self.label_col].iloc[
                               start_index + self.look_back:start_index + self.look_back + self.predict_time_stamp
                               ].reset_index(drop=True).values
                    label.append(sublabel)
                else:
                    subset = dataframe.iloc[start_index:start_index + self.look_back][self.features]
                data.append(subset)
            data_dict['no group'] = data
            if has_label:
                label_dict['no group'] = label
        self.num_features = len(self.features)
        return data_dict, label_dict if has_label else None

    def inverse_transform_labels(self, y_scaled):
        """Reverse the scaling of the label data."""
        y_scaled = np.array(y_scaled)
        if y_scaled.shape==1:
            return self.label_scaler.inverse_transform(y_scaled.reshape(-1, 1))
        else:
            return self.label_scaler.inverse_transform(y_scaled)
    def save_scalers(self, feature_scaler_path, label_scaler_path=None):
        joblib.dump(self.feature_scaler, feature_scaler_path)
        if label_scaler_path and self.label_scaler:
            joblib.dump(self.label_scaler, label_scaler_path)

    def load_scalers(self, feature_scaler_path, label_scaler_path=None):
        self.feature_scaler = joblib.load(feature_scaler_path)
        if label_scaler_path:
            self.label_scaler = joblib.load(label_scaler_path)
