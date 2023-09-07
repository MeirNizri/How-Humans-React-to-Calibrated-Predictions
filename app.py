import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from data_utils.load_data import get_model_prediction
from data_utils.save_data import save_user_data_gspread


# Initialise the Flask app
app = Flask(__name__)
CORS(app)

desired_count = 30
models = ["NN", "NN_PT", "NN_IR", "NN_IR_PT", "NN_IR_PT_Random"]
model_counts = {model: 0 for model in models}

# Function to get 20 random rainfall model forecasts and their true label
@app.route("/<num_days>/")
def get_rainfall(num_days):
    # randomly select one model from models untill all chosen desired number of times
    choose_prob = np.array([(desired_count - model_counts[model]) for model in models])
    choose_prob = choose_prob / np.sum(choose_prob)
    choose_prob = np.nan_to_num(choose_prob, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.sum(choose_prob) != 0:
        model_name = np.random.choice(models, p=choose_prob)
    else:
        model_name = np.random.choice(models)
    
    model_counts[model_name] += 1
    print(model_counts)

    # Get the model prediction and true label
    y_pred, y = get_model_prediction(model_name=model_name, num_samples=int(num_days))

    # Convert the predictions to integers and convert to list
    y_pred = np.rint(y_pred*100).astype(int)
    y_pred = y_pred.tolist()
    y = y.tolist()

    # Send the predictions and true labels as a JSON object
    data = {
        "predictions": y_pred,
        "outcomes": y,
        "model_name": model_name
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
    app.run(host='0.0.0.0', port=5000)
