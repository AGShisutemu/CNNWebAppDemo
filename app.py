# some utilities
import os
import numpy as np
from util import base64_to_pil, save_base64_to_image

# Flask
from flask import (
    Flask,
    redirect,
    url_for,
    request,
    render_template,
    Response,
    jsonify,
    redirect,
    send_from_directory
)

# tensorflow
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import (
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from gradio_client import Client
import json

# Declare a flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000


def classifyImage():
    model = MobileNetV2(weights="imagenet")

    # Loading the pretrained model
    # model_json = open(Model_json, 'r')
    # loaded_model_json = model_json.read()
    # model_json.close()
    # model = model_from_json(loaded_model_json)
    # model.load_weights(Model_weigths)

    return model


def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode="tf")

    preds = model.predict(x)
    return preds


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

app.add_url_rule(
    "/uploads/<name>", endpoint="download_file", build_only=True
)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    client = Client("https://feisarx86-timm-inception-v3-tv-in1k.hf.space/")
    if request.method == "POST":
        save_base64_to_image(request.json, "test_image.jpg")
        print(f"{request.url_root}{UPLOAD_FOLDER}/test_image.jpg")
        result = client.predict(
            f"{request.url_root}{UPLOAD_FOLDER}/test_image.jpg",  # str (filepath or URL to image) in 'Input Image' Image component
            api_name="/predict",
        )
        # # Get the image from post request
        # img = base64_to_pil(request.json)
        # # initialize model
        # model = classifyImage()

        # # Make prediction
        # preds = model_predict(img, model)

        # pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        # result = str(pred_class[0][0][1])               # Convert to string
        # result = result.replace('_', ' ').capitalize()
        # # Serialize the result, you can add additional fields
        try:
            with open(result, "r") as file:
                for line in file:
                    return jsonify(result=json.loads(line.strip()))
        except FileNotFoundError:
            print("File not found.")
        except IOError:
            print("Error reading the file.")
    return None


if __name__ == "__main__":
    # app.run(port=5002)
    app.run()
