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
app.config['UPLOADED_DATAFILES_DEST'] = 'data'  # 设置文件存储位置
configure_uploads(app, data_files)
analysis = DataAnalysis()

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    target_basename = 'train_data'  # Standardized basename for training data

    if 'datafile' in request.files:
        file = request.files['datafile']
        original_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        target_filename = f"{target_basename}.{original_extension}"  # Construct filename with extension

        # Check if the file with the specific name already exists
        target_path = os.path.join(app.config['UPLOADED_DATAFILES_DEST'], target_filename)
        analysis.load_data(target_path)
        if os.path.exists(target_path):
            return jsonify(message="File already exists, upload skipped.", filename=target_filename)

        # Save the file with the specific filename including its extension
        filename = data_files.save(file, name=target_filename)

        if os.path.exists(os.path.join(app.config['UPLOADED_DATAFILES_DEST'], filename)):
            session['uploaded_file_path'] = os.path.join(app.config['UPLOADED_DATAFILES_DEST'], filename)
            # Assuming 'analysis' is an instance of some class defined to handle data
            return jsonify(message="File uploaded and saved as 'train_data' with original extension.", filename=filename)
        else:
            return jsonify(message="Failed to save file.")

    return jsonify(message="No file uploaded.")

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
    data = request.get_json()
    label = data['label']
    excluded_features = data['excludedFeatures']
    # 将这些设置存储在会话中以便后续使用
    session['label'] = label
    session['excluded_features'] = excluded_features
    analysis.remove_features(excluded_features)
    return jsonify(message="Supervised options set successfully: Label - {}, Excluded Features - {}".format(label, excluded_features))

@app.route('/generate_histogram')
def generate_histogram():
    if os.path.exists('images/histogram.png'):
        pass
    else:
        save_path = 'images/histogram.png'
        analysis.histogram(save_path)  # Generate the histogram
    return send_from_directory('images', 'histogram.png')

@app.route('/generate_scatter')
def generate_scatter():
    if os.path.exists('images/scatter_plot.png'):
        pass
    else:
        label = session['label']
        scatter_path = 'images/scatter_plot.png'
        analysis.scatter(label, scatter_path)
    return send_from_directory('images', 'scatter_plot.png')

@app.route('/generate_correlation')
def generate_correlation():
    if os.path.exists('images/correlation_plot.png'):
        pass
    else:
        label = session['label']
        correlation_path = 'images/correlation_plot.png'
        analysis.correlation(label, correlation_path)
    return send_from_directory('images', 'correlation_plot.png')

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
        'mode': mode
    })
@app.route('/confirm_training', methods=['POST'])
def confirm_training():
    mode = session['mode']
    label = session['label']
    analysis.auto_ml(label,mode)
    return jsonify({'message':'get the best model successfully'})

@app.route('/upload-test-data', methods=['POST'])
def upload_test_data():
    target_basename = 'test_data'  # Standardized basename for test data

    if 'testdatafile' in request.files:
        file = request.files['testdatafile']
        original_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        target_filename = f"{target_basename}.{original_extension}"  # Construct filename with extension

        # Check if the file already exists
        target_path = os.path.join(app.config['UPLOADED_DATAFILES_DEST'], target_filename)
        analysis.load_test_data(target_path)
        if os.path.exists(target_path):
            return jsonify(message="Test file already exists, upload skipped.", filename=target_filename)

        # Save the file with the specific filename including its extension
        filename = data_files.save(file, name=target_filename)

        if os.path.exists(os.path.join(app.config['UPLOADED_DATAFILES_DEST'], filename)):
            session['uploaded_test_file_path'] = os.path.join(app.config['UPLOADED_DATAFILES_DEST'], filename)
            # Assuming 'analysis' is an instance of some class defined to handle data

            return jsonify(message="Test file uploaded and saved as 'test_data' with original extension.", filename=filename)
        else:
            return jsonify(message="Failed to save file.")

    return jsonify(message="No test file uploaded.")


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
