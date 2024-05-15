import pandas as pd
from data_analysis_new import TimeSeriesConfig,TimeSeriesAnalysis
from utlis.preprocessing import TimeSeriesPreprocessor
from AutoMachineLearning import TimeSeriesAutoML,GridSearchTuner
from model.models import LSTMModel,GRUModel,BaseTCNModel,FreTS
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
    time_series_config = TimeSeriesConfig(
        task='nasa',
        timestamp_column='time_in_cycles',
        label='RUL',
        group_by='engine_no',
        excluded_features=['op_setting_3', 'sensor_16', 'sensor_19'],
        path='static/data/NASA/train_data.csv'
    )
    start_date = time_series_config.start
    start_test_date = time_series_config.start_test
    end_date = time_series_config.end

    analysis = TimeSeriesAnalysis(time_series_config)
    analysis.load_data()
    analysis.preprocess_data()

    # analysis.analyze_data()#plot time series, auto-correlation and fft

    # look back set up, look back = 1 can use auto regression task, e.g. arima
    # look back>1 use rnn based method
    look_back = 10
    # adjust to predict the next n time stamp
    predict_time_stamp = 1

    data = analysis.get_partial_data(end=start_test_date)
    preprocessor = TimeSeriesPreprocessor(time_series_config, look_back=look_back,
                                          predict_time_stamp=predict_time_stamp)
    preprocessor.fit(data)
    preprocessed_training_data, preprocessed_training_label = preprocessor.transform(data)

    # should specify the dimension here
    input_dimension, output_dimension = preprocessor.num_features, predict_time_stamp
    auto_ml = TimeSeriesAutoML(time_series_config)

    # auto_ml.add_model('LSTM',LSTMModel(input_dimension,output_dimension))
    # auto_ml.add_model('GRU', GRUModel(input_dimension, output_dimension))
    # auto_ml.add_model('TCN', BaseTCNModel(input_dimension, output_dimension))
    auto_ml.add_model('FreTS', FreTS(look_back,input_dimension, output_dimension))
    auto_ml.add_tuner('GridSearch', GridSearchTuner(num_folds=2))

    auto_ml.run_experiments(preprocessed_training_data, preprocessed_training_label)


    # write different evaluation processes here
    path = 'static/data/NASA/test_data.csv'
    analysis.load_data(path)
    analysis.preprocess_data()

    test_data = analysis.get_partial_data()
    preprocessed_test_data, preprocessed_test_label = preprocessor.transform(test_data)

    for key in preprocessed_test_data.keys():
        preprocessed_test_data[key] = [preprocessed_test_data[key][-1]]
    auto_ml.evaluate(preprocessed_test_data, preprocessed_test_label)
    test_result = auto_ml.test_result['FreTS']['test_prediction']
    test_result = preprocessor.inverse_transform_labels(test_result)
    test_result = [int(i) for i in test_result]
    test_result = [1 if i < 100 else 0 for i in test_result]
    data = {
        'engine_no': list(range(len(test_result))),
        'result': test_result
    }

    df = pd.DataFrame(data)
    df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()