# Importing libraries
import pickle
import numpy as np
from flask import Flask, request, render_template

# Create Flask App
app = Flask(__name__)

# Load the pickle file
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("X.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("X.html", prediction_text=" RESULT: {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
    
