# Creating a Flask application with two routes

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from data_utils.load_data import get_model_prediction
from data_utils.save_data import save_user_data_gspread


# Initialise the Flask app
app = Flask(__name__)
CORS(app)


# Function to get 20 random rainfall model forecasts and their true label
@app.route("/<model_name>/")
def get_rainfall(model_name):
    # Get the model prediction and true label
    y_pred, y = get_model_prediction(model_name=model_name, num_samples=20)

    # Convert the predictions to integers and convert to list
    y_pred = np.rint(y_pred*100).astype(int)
    y_pred = y_pred.tolist()
    y = y.tolist()

    # Send the predictions and true labels as a JSON object
    data = {
        "predictions": y_pred,
        "outcomes": y
    }
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# Function to receive an array of data and save it to a gspread sheet
@app.route("/addUser/", methods=["POST"])
def add_user():
    # Get the data from the POST request and save it to a gspread sheet
    data = request.get_json()
    save_user_data_gspread(data)

    # Send a success message back to the client
    response = jsonify({'status': 'success'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
    

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
