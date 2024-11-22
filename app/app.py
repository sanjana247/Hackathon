from flask import Flask, request, jsonify
import joblib
from preprocess import preprocess_text

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('description', '')
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)
    return jsonify({'category': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
