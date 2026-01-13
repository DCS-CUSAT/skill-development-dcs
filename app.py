from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('models/diabetes_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bloodpressure = int(request.form['bloodpressure'])
        bmi = float(request.form['bmi'])
        age = int(request.form['age'])

        input_data = np.array([[pregnancies, glucose, bloodpressure, bmi, age]])
        prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)


# Run the app
if __name__ == '__main__':
    app.run(debug=False)