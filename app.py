# from flask import Flask, request, jsonify
# import pickle
# import numpy as np
# import warnings

# warnings.filterwarnings('ignore')

# model = pickle.load(open('model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "<h1>Hello World!</h1>"

# @app.route('/predict', methods=['POST'])
# def predict():
#     Pregnancies = int(request.form.get('Pregnancies'))
#     Glucose = int(request.form.get('Glucose'))
#     BloodPressure = int(request.form.get('BloodPressure'))
#     SkinThickness = int(request.form.get('SkinThickness'))
#     Insulin = int(request.form.get('Insulin'))
#     BMI = float(request.form.get('BMI'))
#     DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
#     Age = int(request.form.get('Age'))

#     input_query = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
#     input_query_std = scaler.transform(input_query)
    
#     result = model.predict(input_query_std)[0]

#     # Serialize response data into JSON format and set Content-Type header
#     response = jsonify({'Outcome': str(result)})
#     response.headers.add('Content-Type', 'application/json')




# ###################################

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import warnings

warnings.filterwarnings('ignore')

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

# Serve static HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    Pregnancies = int(request.form.get('Pregnancies'))
    Glucose = int(request.form.get('Glucose'))
    BloodPressure = int(request.form.get('BloodPressure'))
    SkinThickness = int(request.form.get('SkinThickness'))
    Insulin = int(request.form.get('Insulin'))
    BMI = float(request.form.get('BMI'))
    DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
    Age = int(request.form.get('Age'))

    input_query = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    input_query_std = scaler.transform(input_query)

    result = model.predict(input_query_std)[0]

    return jsonify({'Outcome': str(result)})

# if __name__ == '__main__':
#     app.run()

#     return response

# if __name__ == "__main__":
#     app.run(debug=True)
