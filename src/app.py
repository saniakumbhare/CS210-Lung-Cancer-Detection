from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Load the model
with open('../models/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame(data, index=[0])
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
