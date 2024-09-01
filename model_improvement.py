import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import pickle
import os

app = Flask(__name__)

# Load the model and handle potential errors
model_path = "../ufo-model.pkl"

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract and validate input data
        form_values = request.form.values()
        if not form_values:
            raise ValueError("No input data provided")

        int_features = [int(x) for x in form_values]
        if len(int_features) == 0:
            raise ValueError("Input data is empty")

        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        
        # Ensure prediction output is valid
        if not isinstance(prediction, (np.ndarray, list)) or len(prediction) != 1:
            raise ValueError("Invalid prediction output")

        output = prediction[0]
        countries = ["Australia", "Canada", "Germany", "UK", "US"]

        # Ensure prediction is within expected range
        if output < 0 or output >= len(countries):
            raise ValueError("Prediction output is out of range")

        return render_template(
            "index.html",
            prediction_text="Likely country: {}".format(countries[output])
        )

    except ValueError as ve:
        return render_template("index.html", prediction_text="Error: {}".format(str(ve)))
    except Exception as e:
        return render_template("index.html", prediction_text="An error occurred: {}".format(str(e)))


if __name__ == "__main__":
    app.run(debug=True)
