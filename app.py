from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Bidirectional  # ✅ add this line

app = Flask(__name__)

# Upload folder config
UPLOAD_FOLDER = os.path.join('static', 'uploaded')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # ensure folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(
    "inference_model1.h5",
    custom_objects={
        "LSTM": LSTM,
        "Bidirectional": Bidirectional
    },
    compile=False
)

# model = load_model(
#     "trained_model_13_3.h5",
#     custom_objects={
#         "LSTM": LSTM,
#         "Bidirectional": Bidirectional
#     },
#     compile=False
# )






# Constants
alphabets = u"!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
img_w, img_h = 256, 64


# Decode numeric prediction to string
def num_to_label(num_seq):
    return "".join([alphabets[ch] for ch in num_seq if ch != -1])


# Preprocess image for model
def preprocess_for_model(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    resized = cv2.resize(contrast, (img_w, img_h))
    norm = resized.astype(np.float32) / 255.0
    norm = np.expand_dims(norm, axis=-1)
    norm = np.transpose(norm, (1, 0, 2))
    return norm


# Decode model output
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded = K.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    out = K.get_value(decoded)
    return num_to_label(out[0])


@app.route('/')
def index():
    return render_template('index.html')  # optional, can create a simple upload form later


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        processed = preprocess_for_model(image)
        processed = np.expand_dims(processed, axis=0)
        prediction = model.predict(processed)
        text = decode_prediction(prediction)

        return jsonify({'text': text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
