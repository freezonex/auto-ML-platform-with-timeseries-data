import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from auto_machine_learning import AUTOML,AUTOML_for_time_series
from sklearn.metrics import mean_squared_error, f1_score,confusion_matrix
import numpy as np
from statsmodels.tsa.stattools import acf
class DataAnalysis:
    def __init__(self, dataframe=None):
        self.dataframe = dataframe
        self.cur_df = None
        self.label = None
        self.x_scaler = None
        self.test_df = None
        self.best_model = None
        self.group_by = None
    def load_data(self,filepath):
        self.df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        self.cur_df = self.df
    def load_test_data(self,filepath):
        self.test_df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)

    def remove_nan(self):
        initial_columns = set(self.df.columns)
        self.cur_df.dropna(axis=1, how='all', inplace=True)
        cleaned_columns = set(self.cur_df.columns)
        removed_columns = initial_columns - cleaned_columns
        return removed_columns

    def remove_features(self,excluded_features):
        if excluded_features:
            # Create a list of features to drop by checking if they exist in the DataFrame's columns
            features_to_drop = [feature for feature in excluded_features if feature in self.cur_df.columns]
            if features_to_drop:
                # Drop the columns that exist in the DataFrame
                self.cur_df = self.cur_df.drop(columns=features_to_drop)
            else:
                print("No valid columns were provided to remove.")
        else:
            pass

    def histogram(self,savepath,is_time_series=False):
        if not is_time_series:
            num_features = len(self.cur_df.columns)
            num_rows = (num_features + 1) // 2  # Adjust the number of rows in the grid
            plt.figure(figsize=(10, num_rows * 5))  # Adjust the figure size appropriately
            for i, column in enumerate(self.cur_df.columns):
                plt.subplot(num_rows, 2, i + 1)  # Create subplots in a grid of 'num_rows x 2'
                self.cur_df[column].hist(bins='auto', alpha=0.75)
                plt.title(f'Histogram of {column}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(savepath)
            plt.close()
        else:
            groups = self.cur_df[self.group_by].unique()
            features = [col for col in self.cur_df.columns if col not in ['date', self.group_by, self.label]]
            num_groups = len(groups)
            num_features = len(features)
            fig, axes = plt.subplots(nrows=num_groups, ncols=num_features, figsize=(15, num_groups * 4))
            axes = np.atleast_2d(axes).transpose()

            for i, group in enumerate(groups):
                group_data = self.cur_df[self.cur_df[self.group_by] == group]
                for j, feature in enumerate(features):
                    ax = axes[i, j]
                    ax.plot(group_data['date'], group_data[feature], marker='o', linestyle='-')
                    ax.set_title(f'{group} - {feature}')
                    ax.set_ylabel('Value')
                    if i == num_groups - 1:
                        ax.set_xlabel('Date')
                    if j == 0:
                        ax.set_ylabel('Value')

                # 设置图表布局
            plt.tight_layout()
            plt.savefig(savepath)
            plt.close()

    def scatter(self,label,savepath,is_time_series=False):
        if not is_time_series:
            if label in self.cur_df.columns:
                features = [col for col in self.cur_df.columns if col != label]
                num_features = len(features)
                num_rows = (num_features + 1) // 2  # Adjust the number of rows in the grid
                plt.figure(figsize=(10, num_rows * 5))
                for i, feature in enumerate(features):
                    plt.subplot(num_rows, 2, i + 1)
                    plt.scatter(self.cur_df[feature], self.cur_df[label], alpha=0.5)
                    plt.title(f'Scatter Plot of {feature} vs. {label}')
                    plt.xlabel(feature)
                    plt.ylabel(label)
                plt.tight_layout()
                plt.savefig(savepath)
                plt.close()
        else:
            groups = self.cur_df[self.group_by].unique()
            features = [col for col in self.cur_df.columns if col not in ['date', self.group_by, self.label]]
            num_groups = len(groups)
            num_features = len(features)
            fig, axes = plt.subplots(nrows=num_groups, ncols=num_features, figsize=(15, num_groups * 4))
            axes = np.atleast_2d(axes).transpose()

            for i, group in enumerate(groups):
                group_data = self.cur_df[self.cur_df[self.group_by] == group]
                for j, feature in enumerate(features):
                    ax = axes[i, j]
                    # Calculate autocorrelation
                    feature_data = group_data[feature].dropna()
                    lag_acf = acf(feature_data, nlags=10)
                    ax.stem(range(len(lag_acf)), lag_acf, basefmt="b-", use_line_collection=True)
                    ax.set_title(f'{group} - {feature} Autocorrelation')
                    ax.set_ylabel('Autocorrelation')
                    if i == num_groups - 1:
                        ax.set_xlabel('Lags')
                    if j == 0:
                        ax.set_ylabel('Autocorrelation')

            # Set the layout and save the figure
            plt.tight_layout()
            plt.savefig(savepath)
            plt.close()

    def correlation(self, label, savepath):

        # Filter out the label from the features and calculate correlation with the label
        features = self.cur_df.columns.drop(label)
        correlation_data = self.cur_df[features].corrwith(self.cur_df[label])

        # Plotting the correlation values
        plt.figure(figsize=(8, 6))
        sns.barplot(x=correlation_data.values, y=correlation_data.index)
        plt.title('Feature Correlation with ' + label)
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()

    def auto_ml(self,label,mode,time_series):
        print('start training')
        if not time_series:
            features = [col for col in self.cur_df.columns if col != label]
            x = self.cur_df[features]
            y = self.cur_df[label]
            self.best_model,self.x_scaler = AUTOML(x, y, mode)
        else:
            AUTOML_for_time_series(self.cur_df,label,group_by=self.group_by)






    def evaluation(self,label,mode,threshold=None):
        feature_name = [col for col in self.cur_df.columns if col != label]

        x = self.test_df[feature_name]
        x_scaled = self.x_scaler.transform(x)
        y = self.test_df[label]

        prediction = self.best_model.predict(x_scaled)


        if mode == 'regression':
            test_score = mean_squared_error(y, prediction, squared=False)

            # Create scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y, prediction, alpha=0.5, color='blue', label='Predicted vs Actual')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs. Predicted Values Scatter Plot, test score:{test_score}')
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2,
                     label='Ideal Fit')  # Ideal line where y = prediction
            plt.legend()
            plt.grid(True)
            plt.savefig('static/result_images/scatter.png')
            plt.close()

            # Get feature importances from the model
            importances = self.best_model.feature_importances_

            # Create a series with feature names and their importance scores
            importance_dict = dict(zip(feature_name, importances))
            importance_series = pd.Series(importance_dict).sort_values(ascending=False)

            # Plotting
            plt.figure(figsize=(10, 6))
            importance_series.plot(kind='bar', color='skyblue')
            plt.title('Feature Importances')
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.xticks(rotation=45)
            plt.savefig('static/result_images/importance.png')
            plt.close()

            if threshold:
                y_converted = [1 if truth < threshold else 0 for truth in y]
                prediction_converted = [1 if truth < threshold else 0 for truth in prediction]

                f1 = f1_score(y_converted,prediction_converted)
                cm = confusion_matrix(y_converted, prediction_converted)

                # Plot the confusion matrix using seaborn
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'],
                            yticklabels=['False', 'True'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Confusion Matrix,f1 score:{f1}')
                plt.savefig('static/result_images/confusion_matrix.png')
                plt.close()




if __name__ == '__main__':
    analysis = DataAnalysis()
    analysis.load_data(filepath='/mnt/ylq/liqiang_old_ubuntu/liqiang/PycharmProjects/predictiveMaintainance/data/train_data.csv')
    analysis.remove_nan()
    analysis.remove_features(['engine_no','time_in_cycles'])
    analysis.histogram('images/histogram.png')