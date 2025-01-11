from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])

        # Make predictions
        prediction = model.predict(input_df)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
