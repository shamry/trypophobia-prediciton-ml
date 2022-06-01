import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from ml.prediction import predict

app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Controller
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict_controller():
    if 'frame' not in request.files:
        return jsonify({
            "response": "UI Layout required"
        }), 400
    file = request.files['frame']
    if file.filename == '':
        return jsonify({
            "response": "UI Layout required"
        }), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        result = predict(file_path)
        return jsonify({
            "response": result
        }), 200
    else:
        return jsonify({
            "response": "This UI format not accepted"
        }), 400


if __name__ == '__main__':
    app.run()
