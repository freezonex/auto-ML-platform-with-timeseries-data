# Auto-Machine Learning for Predictive Maintenance

## Getting Started

Run `app.py` to launch the predictive maintenance application.

### Prerequisites

Before you begin, ensure you have the following packages installed:

- Flask
- Matplotlib
- Pandas
- Seaborn
- scikit-learn
- XGBoost
- Joblib

Install these packages using pip:

```bash
pip install flask matplotlib pandas seaborn scikit-learn xgboost joblib
```

### Step-by-Step Instructions

1. **Upload Training Data**
   - Upload your training data through the application interface.

2. **Display Data**
   - Verify the uploaded data by viewing the head of the dataframe displayed on the interface.

3. **Start Data Analysis**
   - Click 'Start Data Analysis' to remove columns with NaN values, preparing the data for machine learning.

4. **Supervised Learning Support**
   - This application supports supervised learning only.

5. **Specify Configuration**
   - Designate 'RUL' as the label for prediction.
   - Exclude the following features to optimize model performance:
     - `engine_no`
     - `time_in_cycles`
     - `op_setting_3`
     - `sensor_16`
     - `sensor_19`

6. **Visualization and AutoML**
   - Access visualization options in the interface.
   - Select 'regression' and initiate the auto machine learning process.

7. **Training and Testing**
   - Confirm settings and start the training process.
   - After training, upload the test data for evaluation.
   - Set the RUL threshold to 100 as per the official competition guidelines.

8. **Review Results**
   - Results can be viewed via a scatter plot.
   - Model performance can be assessed using the f1 score.
   - Feature analysis is available to understand feature importance.