import os.path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from auto_machine_learning import AUTOML,AUTOML_for_time_series
from sklearn.metrics import mean_squared_error, f1_score,confusion_matrix
from statsmodels.tsa.stattools import acf
from abc import ABC, abstractmethod
from copy import deepcopy
from statsmodels.tsa.seasonal import seasonal_decompose
from utlis.plot import TimeSeriesPlot
from utlis.preprocessing import TimeSeriesPreprocessor
class BaseConfig:
    def __init__(self, task,group_by=None, label=None, excluded_features=None):
        self.group_by = group_by
        self.label = label
        self.excluded_features = excluded_features or []
        self.task = task
    def update_excluded_features(self, new_excluded_features):
        self.excluded_features.update(new_excluded_features)
class TimeSeriesConfig(BaseConfig):
    def __init__(self, timestamp_column, resample_rule=None, **kwargs):
        super().__init__(**kwargs)
        self.timestamp_column = timestamp_column
        self.resample_rule = resample_rule  # 可选，用于定义重采样规则
class DataAnalysisInterface(ABC):
    def __init__(self,config):
        self.config = config
        self.dataframe = None
        self.processed_dataframe = None

    def load_data(self, filepath: str):
        self.dataframe = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        self.processed_dataframe = deepcopy(self.dataframe)
        print(f'Loading data from {filepath}')

    @abstractmethod
    def preprocess_data(self):
        self.processed_dataframe.dropna(axis=1,how='all',inplace=True)

        if self.config.excluded_features:
            self.processed_dataframe.drop(column=self.config.excluded_features,inplace=True,errors='ignore')

    @abstractmethod
    def analyze_data(self):
        """Perform data analysis."""
        pass

    def update_config(self,new_excluded_features):
        self.config.update_excluded_features(new_excluded_features)
        print("Config updated with new excluded features:", self.config.excluded_features)
        self.preprocess_data()  # 重新预处理数据
        self.analyze_data()  # 重新分析数据
class TimeSeriesAnalysis(DataAnalysisInterface):
    def preprocess_data(self):
        super().preprocess_data()
        # 设置时间索引
        if self.config.timestamp_column in self.processed_dataframe.columns:
            self.processed_dataframe[self.config.timestamp_column] = pd.to_datetime(
                self.processed_dataframe[self.config.timestamp_column])
            self.processed_dataframe.set_index(self.config.timestamp_column, inplace=True)
            print(f"Timestamp column '{self.config.timestamp_column}' set as index.")

        # 应用分组和填充逻辑
        if self.config.group_by and self.config.group_by in self.processed_dataframe.columns:
            # 分组并对每个组进行操作
            grouped = self.processed_dataframe.groupby(self.config.group_by)
            # 用相邻的时间戳填充每个组的缺失值
            self.processed_dataframe = grouped.apply(lambda group: group.ffill().bfill())
            print("Data grouped by", self.config.group_by, "and missing values filled.")
        else:
            # 没有指定分组，直接在整个数据集上进行操作
            self.processed_dataframe.ffill(inplace=True)  # 先前填充
            self.processed_dataframe.bfill(inplace=True)  # 后向填充
            print("Missing values filled based on time adjacency without grouping.")

    def analyze_data(self):
        analyzer = TimeSeriesPlot(self.config, self.processed_dataframe)
        analyzer.analyze_data()

if __name__ == '__main__':
    path = 'static/data/warehouse/train_data.csv'
    time_series_config = TimeSeriesConfig(
        task = 'warehouse',
        timestamp_column='date',
        label='next_day_storage',
        group_by='warehouse name'
    )
    analysis = TimeSeriesAnalysis(time_series_config)
    analysis.load_data(path)
    analysis.preprocess_data()
    print(analysis.processed_dataframe.head())
    # analysis.analyze_data()
    data = analysis.processed_dataframe
    preprocessor = TimeSeriesPreprocessor(time_series_config,look_back=3)
    preprocessor.fit(data)
    preprocessed_training_data = preprocessor.transform(data)

    print(preprocessed_training_data)