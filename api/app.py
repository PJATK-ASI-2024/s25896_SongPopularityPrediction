from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Loading trained model
MODEL_PATH = "model.pkl"
try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise Exception(f"Model file not found at {MODEL_PATH}. Make sure the file exists.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parsing input JSON
        input_data = request.get_json()
        
        # Converting JSON to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Checking if required columns are present
        expected_columns = ['Energy', 'Dancebility', 'Loudness']
        if not all(col in input_df.columns for col in expected_columns):
            return jsonify({"error": f"Missing required columns. Expected: {expected_columns}"}), 400

        # Makeing predictions
        prediction = model.predict(input_df)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
