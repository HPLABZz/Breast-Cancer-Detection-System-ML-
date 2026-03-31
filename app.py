from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']

        if len(data) != 30:
            return jsonify({"error": "Exactly 30 csv values required"}), 400

        input_array = np.array(data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]

        result = "Malignant (Cancerous Cell Present)" if prediction == 'M' else "Benign (Cancerous Cell  Not Present)"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)