from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sl = float(request.form['sl'])
    sw = float(request.form['sw'])
    pl = float(request.form['pl'])
    pw = float(request.form['pw'])

    data = scaler.transform([[sl, sw, pl, pw]])
    prediction = model.predict(data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
