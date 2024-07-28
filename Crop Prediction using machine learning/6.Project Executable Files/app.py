from flask import Flask, render_template, request
import numpy as np
import pickle

model_path = 'model.pkl'  # Ensure this path is correct

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except (FileNotFoundError, EOFError) as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__, static_url_path='/static')

# Your other routes and functions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def pred():
    return render_template('details.html')

@app.route('/crop_predict', methods=['POST'])
def crop_predict():
    if not model:
        return "Model is not loaded properly. Please check the model file."

    # Get the input data from the form
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Make a prediction using the loaded model
    prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    crop = prediction[0]

    return render_template('crop_predict.html', crop=crop)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
