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
from utlis.plot import TimeSeriesPlot,plot_grouped_time_series
from utlis.preprocessing import TimeSeriesPreprocessor
from AutoMachineLearning import TimeSeriesAutoML,GridSearchTuner
from model.models import LSTMModel,GRUModel,BaseTCNModel
from matplotlib.dates import DateFormatter, MonthLocator
from datetime import datetime, timedelta
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

        columns_to_fill = self.processed_dataframe.columns.difference([self.config.label])

        # Apply group and fill logic
        if self.config.group_by and self.config.group_by in self.processed_dataframe.columns:
            # Group by the specified column and apply filling only to the specified columns to fill
            grouped = self.processed_dataframe.groupby(self.config.group_by)
            self.processed_dataframe[columns_to_fill] = grouped[columns_to_fill].apply(
                lambda group: group.ffill().bfill())
            print("Data grouped by", self.config.group_by, "and missing values in features filled.")
        else:
            # Apply filling to the entire dataset excluding the label column
            self.processed_dataframe[columns_to_fill] = self.processed_dataframe[columns_to_fill].ffill().bfill()
            print("Missing values in features filled based on time adjacency without grouping.")

    def get_partial_data(self, start=None, end=None):
        # Slicing directly using the datetime index if start and/or end are provided
        if start and end:
            return self.processed_dataframe.loc[start:end].copy()
        elif start:
            return self.processed_dataframe.loc[start:].copy()
        elif end:
            return self.processed_dataframe.loc[:end].copy()
        else:
            return self.processed_dataframe.copy()

    def analyze_data(self):
        analyzer = TimeSeriesPlot(self.config, self.processed_dataframe)
        analyzer.analyze_data()

if __name__ == '__main__':
    path = 'static/data/warehouse/train_data.csv'
    time_series_config = TimeSeriesConfig(
        task = 'warehouse',
        timestamp_column='date',
        label='next_day_storage',
        group_by='warehouse_name'
    )
    start_date = '2023-08-01'
    start_test_date = '2023-11-07'
    end_date = '2023-11-30'


    analysis = TimeSeriesAnalysis(time_series_config)
    analysis.load_data(path)
    analysis.preprocess_data()
    print(analysis.processed_dataframe.head())
    # analysis.analyze_data()

    #look back set up, look back = 1 can use auto regression task, e.g. arima
    #look back>1 use rnn based method
    look_back = 3

    data = analysis.get_partial_data(start_date,start_test_date)
    preprocessor = TimeSeriesPreprocessor(time_series_config, look_back=look_back)
    preprocessor.fit(data)
    preprocessed_training_data, preprocessed_training_label = preprocessor.transform(data)

    # test_path = 'static/data/warehouse/test_data.csv'
    #define the test data start from which date,consider the look back
    start_test_date = datetime.strptime(start_test_date, '%Y-%m-%d')
    start_test_date = start_test_date+timedelta(days=1)-timedelta(days=look_back)
    start_test_date = start_test_date.strftime('%Y-%m-%d')

    #read and process again, since new data is coming
    analysis.load_data(path)
    analysis.preprocess_data()
    # analysis.preprocess_data()
    print(analysis.processed_dataframe.head())
    test_data = analysis.get_partial_data(start_test_date,end_date)
    preprocessed_test_data, preprocessed_test_label = preprocessor.transform(test_data)

    #should specify the dimension here
    input_dimension,output_dimension = 1,1
    auto_ml = TimeSeriesAutoML()

    auto_ml.add_model('LSTM',LSTMModel(input_dimension,output_dimension))
    auto_ml.add_model('GRU',GRUModel(input_dimension,output_dimension))
    auto_ml.add_model('TCN',BaseTCNModel(input_dimension,output_dimension))
    auto_ml.add_tuner('GridSearch',GridSearchTuner(num_folds=2))

    auto_ml.run_experiments(preprocessed_training_data,preprocessed_training_label)
    #result includes train_loss, test_loss, train_prediction, test_prediction
    results = auto_ml.evaluate(preprocessed_test_data, preprocessed_test_label)

    for model_name,result in results.items():
        train_prediction = result['train_prediction']
        train_prediction = preprocessor.inverse_transform_labels(train_prediction)

        test_prediction = result['test_prediction']
        test_prediction = preprocessor.inverse_transform_labels(test_prediction)

        num_groups = len(preprocessed_training_label)
        fig, axes = plt.subplots(nrows=num_groups, ncols=1, figsize=(10, 5 * num_groups))

        # Define the overall date range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Generate a date range for plotting
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        start_index_train = 0
        start_index_test = 0
        for i, (name, truth) in enumerate(preprocessed_training_label.items()):
            train_truth = preprocessor.inverse_transform_labels(np.array(truth))
            test_truth = preprocessor.inverse_transform_labels(np.array(preprocessed_test_label[name]))

            num_train = len(train_truth)
            num_test = len(test_truth)

            current_train_pred = train_prediction[start_index_train:start_index_train+num_train]
            current_test_pred = test_prediction[start_index_test:start_index_test+num_test]
            start_index_train += num_train
            start_index_test += num_test

            # Plotting
            ax = axes[i]
            ax.plot(date_range[:num_train], train_truth, label='Train Truth', color='blue')
            ax.plot(date_range[:num_train], current_train_pred, label='Train Prediction', color='red', linestyle='--')
            ax.plot(date_range[num_train:num_train + num_test], test_truth, label='Test Truth', color='green')
            ax.plot(date_range[num_train:num_train + num_test], current_test_pred, label='Test Prediction',
                    color='purple', linestyle='--')

            # Formatting the x-axis
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            ax.set_title(f'Group: {name}')
            ax.legend()
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
        plt.title(f"{model_name}:{results[model_name]['test_loss']}")
        plt.tight_layout()
        plt.savefig(f'{model_name}test.png')