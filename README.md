steps for auto-machine learning for predictive maintenance 
run app.py
1. upload train data
2. display data to view the current dataframe head
3. click start data analysis (will remove nan columns)
4. only support supervised learning
5. specify label in this work is RUL, excluded features (engine_no,time_in_cycles,op_setting_3,sensor_16,sensor_19) recommended
6. click visualization options, then click start auto machine learning, then choose regression
7. confirm and start training
8. start testing
9. upload test data
10. set RUL treshold to 100 as suggested by official competition
11. can see the result via scatter plot, f1 score and feature analysis

required packages: flask,matplotlib,pandas,seaborn,scikit-learn,xgboost,joblib
