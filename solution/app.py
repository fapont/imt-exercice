import pickle

import pandas as pd

from flask import Flask, render_template, request

from utils import transform_house_type


app = Flask(__name__)


@app.route("/hello", methods=["GET"])
def hello_world():
    return "<h1>Hello world</h1>"


@app.route("/app", methods=["GET"])
def render_app():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Load data and transform "house_type" feature
    data = dict(request.form)
    data["house_type"] = transform_house_type(data["house_type"])

    # Convert dictionnary to pandas series
    data = pd.Series(data, dtype=float)

    # Load the model
    model = pickle.load(open("model.pkl", "rb"))

    # Predict using model and return value
    value = round(model.predict(data.values.reshape(1, -1))[0])
    return f"{value} €"


if __name__ == "__main__":
    app.run(port=5678, host="0.0.0.0", debug=True)
