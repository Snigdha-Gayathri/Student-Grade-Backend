from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "✅ Student Grade Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    studytime = int(data["studytime"])
    absences = int(data["absences"])
    failures = int(data["failures"])

    # Model expects a 2D array
    prediction = model.predict([[studytime, absences, failures]])[0]

    return jsonify({"grade": str(prediction)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
