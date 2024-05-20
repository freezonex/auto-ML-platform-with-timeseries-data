import time
import numpy as np
import pandas as pd
from data_analysis_new import TimeSeriesConfig,TimeSeriesAnalysis
from utlis.preprocessing import TimeSeriesPreprocessor
from AutoMachineLearning import TimeSeriesAutoML,GridSearchTuner
from model.models import LSTMModel,GRUModel,BaseTCNModel,FreTS
from model.iTransformer import ITransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime,timedelta
from matplotlib import pyplot as plt
def main():
    # time_series_config = TimeSeriesConfig(
    #     task = 'warehouse',
    #     timestamp_column='date',
    #     label='next_day_storage',
    #     group_by='warehouse_name',
    #     start='2023-08-01',
    #     start_test='2023-11-07',
    #     end='2023-11-30',
    #     path='static/data/warehouse/train_data.csv'
    # )
    # time_series_config = TimeSeriesConfig(
    #     task='nasa',
    #     timestamp_column='time_in_cycles',
    #     label='RUL',
    #     group_by='engine_no',
    #     excluded_features=['op_setting_3', 'sensor_16', 'sensor_19'],
    #     path='static/data/NASA/train_data.csv'
    # )
    time_series_config = TimeSeriesConfig(
        task='ETT',
        timestamp_column='date',
        label='OT',
        path='static/data/ETdataset/ETDataset/ETTh1.csv',
        start = '2016-06-01 00:00:00',
        start_test='2018-04-01 00:00:00'
    )

    start = time_series_config.start
    start_test = time_series_config.start_test
    end = time_series_config.end

    analysis = TimeSeriesAnalysis(time_series_config)
    analysis.load_data()
    analysis.preprocess_data()

    # analysis.analyze_data()#plot time series, auto-correlation and fft

    # look back set up, look back = 1 can use auto regression task, e.g. arima
    # look back>1 use rnn based method
    look_back = 96
    # adjust to predict the next n time stamp
    predict_time_stamp = 96 # 96,192,336,720 in paper

    data = analysis.get_partial_data(end=start_test)
    preprocessor = TimeSeriesPreprocessor(time_series_config, look_back=look_back,
                                          predict_time_stamp=predict_time_stamp)
    preprocessor.fit(data)
    preprocessed_training_data, preprocessed_training_label = preprocessor.transform(data)

    # should specify the dimension here
    input_dimension, output_dimension = preprocessor.num_features, predict_time_stamp
    auto_ml = TimeSeriesAutoML(time_series_config)
    #
    # auto_ml.add_model('LSTM',LSTMModel(input_dimension,output_dimension))
    # # auto_ml.add_model('GRU', GRUModel(input_dimension, output_dimension))
    # auto_ml.add_model('FreTS', FreTS(look_back,input_dimension, output_dimension))
    # auto_ml.add_model('FreTS', FreTS(look_back,input_dimension, output_dimension))
    auto_ml.add_model('ITransformer', ITransformer(input_dimension, output_dimension,look_back,pred_len=predict_time_stamp))
    auto_ml.add_tuner('GridSearch', GridSearchTuner(num_folds=2))
    #
    auto_ml.run_experiments(preprocessed_training_data, preprocessed_training_label)

    # train_result = auto_ml.train_result['FreTS']['train_prediction']

    test_data = analysis.get_partial_data(start=start_test)

    preprocessed_test_data, preprocessed_test_label = preprocessor.transform(test_data)
    auto_ml.evaluate(preprocessed_test_data, preprocessed_test_label)
    test_result = auto_ml.test_result['ITransformer']['test_prediction']
    test_result = preprocessor.inverse_transform_labels(test_result)


    all_truths = []
    for name,truth in preprocessed_test_label.items():
        test_truth = preprocessor.inverse_transform_labels(np.array(truth))
        # input_data = preprocessed_test_data[name]
        # for j,test_label in enumerate(test_truth):
        #     cur_data = np.array(input_data[j])
        #     cur_data = cur_data[:,-1:]
        #     cur_data = preprocessor.inverse_transform_labels(cur_data)
        #     a = np.concatenate((cur_data,test_label.reshape(-1,1)),axis=0)
        #     b = np.concatenate((cur_data,test_result[j].reshape(-1,1)),axis=0)
        #     plot_start_time = pd.to_datetime(start_test)-timedelta(hours=look_back-j-1)
        #     plot_end_time = pd.to_datetime(start_test)+timedelta(hours=look_back+j)
        #     date_range = pd.date_range(start=plot_start_time, end=plot_end_time, freq='H')
        #     plt.plot(date_range,b,label='predict')
        #     plt.plot(date_range, a, label='truth')
        #     plt.legend()
        #     plt.show()
        #     time.sleep(2)
        #     plt.close()


        all_truths.extend(test_truth)
    mse = mean_squared_error(all_truths, test_result)
    mae = mean_absolute_error(all_truths, test_result)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")


    # write different evaluation processes here
    # path = 'static/data/NASA/test_data.csv'
    # analysis.load_data(path)
    # analysis.preprocess_data()
    #
    # test_data = analysis.get_partial_data()
    # preprocessed_test_data, preprocessed_test_label = preprocessor.transform(test_data)
    #
    # for key in preprocessed_test_data.keys():
    #     preprocessed_test_data[key] = [preprocessed_test_data[key][-1]]
    # auto_ml.evaluate(preprocessed_test_data, preprocessed_test_label)
    # test_result = auto_ml.test_result['LSTM']['test_prediction']
    # test_result = preprocessor.inverse_transform_labels(test_result)
    # test_result = [int(i) for i in test_result]
    # test_result = [1 if i < 100 else 0 for i in test_result]
    # data = {
    #     'engine_no': list(range(len(test_result))),
    #     'result': test_result
    # }
    #
    # df = pd.DataFrame(data)
    # df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()