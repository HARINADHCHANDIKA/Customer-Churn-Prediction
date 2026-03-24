from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model/churn_model.pkl')

@app.route('/')
def home():
    return "Churn Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([list(data.values())])
    return jsonify({'churn_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
