from flask import Flask, request, jsonify, render_template, session,send_from_directory
from flask_uploads import UploadSet, configure_uploads, DATA
import pandas as pd
import os
from data_analysis import DataAnalysis
from io import BytesIO
import matplotlib.pyplot as plt
import pickle

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# 配置文件上传
data_files = UploadSet('datafiles', DATA)
app.config['UPLOADED_DATAFILES_DEST'] = 'static/data'  # 设置文件存储位置
configure_uploads(app, data_files)
analysis = DataAnalysis()

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'datafile' not in request.files:
        return jsonify(message="No file uploaded.")
    taskname = request.form['taskname']
    target_directory = os.path.join(app.config['UPLOADED_DATAFILES_DEST'], taskname)
    session['taskname'] = taskname

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    target_basename = 'train_data'  # Standardized basename for training data

    file = request.files['datafile']
    original_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    target_filename = f"{target_basename}.{original_extension}"  # Construct filename with extension

    # Check if the file with the specific name already exists
    target_path = os.path.join(target_directory, target_filename)

    if os.path.exists(target_path):
        analysis.load_data(target_path)
        return jsonify(message="File already exists, upload skipped.", filename=target_filename)

    session['train_data_path'] = target_path

    file.save(target_path)
    analysis.load_data(target_path)
    return jsonify(message="File uploaded and saved in the task-specific directory.", filename=target_filename)


@app.route('/display-data', methods=['GET'])
def display_data():
    if analysis.cur_df is not None:

        return analysis.cur_df.head().to_html(classes='data', border=0)
    else:
        return "No file uploaded"

@app.route('/pre-analyze', methods=['GET'])
def pre_analyze():
    if analysis.cur_df is not None:
        removed_columns = analysis.remove_nan()
        message = "No columns are all NaN." if not removed_columns else f"Columns removed because all values are NaN: {', '.join(removed_columns)}"
        session['nan_columns'] = list(removed_columns)
        return jsonify({"message": message})
    else:
        return "No file uploaded"

@app.route('/set-supervised-options', methods=['POST'])
def set_supervised_options():
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    if not os.path.exists(os.path.join('static/images',session['taskname'])):
        os.makedirs(os.path.join('static/images',session['taskname']))
    data = request.get_json()
    label = data['label']
    excluded_features = data['excludedFeatures']
    is_time_series = data['isTimeSeries'] == 'true'
    group_by = data.get('groupBy', None)

    # 将这些设置存储在会话中以便后续使用
    session['label'] = label
    session['excluded_features'] = excluded_features
    session['is_time_series'] = is_time_series
    session['group_by'] = group_by

    analysis.group_by = group_by
    analysis.label = label
    analysis.remove_features(excluded_features)
    return jsonify(
        message=f"Supervised options set successfully: Label - {label}, Excluded Features - {excluded_features}, Is Time-Series - {is_time_series}")

@app.route('/generate_histogram')
def generate_histogram():
    if session['is_time_series']:
        if os.path.exists(os.path.join('static/images',session['taskname'],'time-series.png')):
            pass
        else:
            save_path = os.path.join('static/images',session['taskname'],'time-series.png')
            analysis.histogram(save_path,session['is_time_series'])  # Generate the histogram
        return send_from_directory(os.path.join('static/images',session['taskname']), 'time-series.png')
    else:
        if os.path.exists(os.path.join('static/images',session['taskname'],'histogram.png')):
            pass
        else:
            save_path = os.path.join('static/images',session['taskname'],'histogram.png')
            analysis.histogram(save_path)  # Generate the histogram
        return send_from_directory(os.path.join('static/images',session['taskname']), 'histogram.png')

@app.route('/generate_scatter')
def generate_scatter():
    if session['is_time_series']:
        if os.path.exists(os.path.join('static/images',session['taskname'],'auto-correlation.png')):
            pass
        else:
            save_path = os.path.join('static/images',session['taskname'],'auto-correlation.png')
            analysis.scatter(None, save_path,session['is_time_series'])  # Generate the histogram
        return send_from_directory(os.path.join('static/images',session['taskname']), 'auto-correlation.png')
    else:
        if os.path.exists(os.path.join('static/images',session['taskname'],'scatter.png')):
            pass
        else:
            label = session['label']
            scatter_path = os.path.join('static/images',session['taskname'],'scatter.png')
            analysis.scatter(label, scatter_path)
        return send_from_directory(os.path.join('static/images',session['taskname']), 'scatter.png')


@app.route('/generate_correlation')
def generate_correlation():
    if os.path.exists(os.path.join('static/images',session['taskname'],'correlation.png')):
        pass
    else:
        label = session['label']
        correlation_path = os.path.join('static/images',session['taskname'],'correlation.png')
        analysis.correlation(label, correlation_path)
    return send_from_directory(os.path.join('static/images',session['taskname']), 'correlation.png')

@app.route('/start_ml', methods=['POST'])
def start_ml():
    data = request.get_json()
    mode = data['mode']
    session['mode'] = mode
    label = session['label']
    excluded_features = session['excluded_features']+session['nan_columns']

    return jsonify({
        'message': f'Model training initiated in {mode} mode',
        'label': label,
        'excluded_features': excluded_features,
        'mode': mode,
        'time_series': session['is_time_series']
    })
@app.route('/confirm_training', methods=['POST'])
def confirm_training():
    mode = session['mode']
    label = session['label']
    analysis.auto_ml(label,mode,session['is_time_series'])
    return jsonify({'message':'get the best model successfully'})

@app.route('/upload-test-data', methods=['POST'])
def upload_test_data():
    if 'testdatafile' not in request.files:
        return jsonify(message="No file uploaded.")
    if not os.path.exists('static/result_images'):
        os.makedirs('static/result_images')
    target_basename = 'test_data'  # Standardized basename for test data
    file = request.files['testdatafile']
    original_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    target_filename = f"{target_basename}.{original_extension}"  # Construct filename with extension

    taskname = session['taskname']

    # Check if the file already exists
    target_directory = os.path.join(app.config['UPLOADED_DATAFILES_DEST'], taskname)
    target_path = os.path.join(target_directory, target_filename)

    if os.path.exists(target_path):
        analysis.load_test_data(target_path)
        return jsonify(message="Test file already exists, upload skipped.", filename=target_filename)

    # Save the file with the specific filename including its extension
    file.save(target_path)
    session['uploaded_test_file_path'] = target_path

    return jsonify(message="Test file uploaded.")


@app.route('/evaluate', methods=['POST'])
def evaluate():

    data = request.get_json()
    threshold = data.get('threshold', None)
    if threshold:
        try:
            threshold = float(threshold)
        except ValueError:
            return jsonify(message="Invalid threshold value."), 400
    try:
        # Assuming analysis.evaluation properly handles the case when threshold is None
        analysis.evaluation(session['label'], session['mode'], threshold)
        return jsonify(message='Successful evaluation')
    except Exception as e:
        return jsonify(message=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)