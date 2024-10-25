from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

model = joblib.load('loan_prediction_model.pkl')
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Data received:", data)  # Debug print
    income = data['income']
    credit_score = data['credit_score']
    # Prepare the input
    input_data = np.array([[income, credit_score]])
    # Make prediction
    prediction = model.predict(input_data)
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    # Convert prediction to readable messages
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)