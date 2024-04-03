from flask import Flask,request,jsonify
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open('model.pkl','rb'))
scaler= pickle.load(open('scaler.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return "<html><body><h1>Hello World!</h1></body></html>"

@app.route('/predict',methods=['POST']) # without url inpit
def predict():
    Pregnancies = int(request.form.get('Pregnancies'))
    Glucose = int(request.form.get('Glucose'))
    BloodPressure = int(request.form.get('BloodPressure'))
    SkinThickness = int(request.form.get('SkinThickness'))
    Insulin = int(request.form.get('Insulin'))
    BMI = float(request.form.get('BMI'))
    DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
    Age = int(request.form.get('Age'))


    input_query=np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    input_query_std = scaler.transform(input_query)
    
    result= model.predict(input_query_std)[0]

    return jsonify({'Outcome':str(result)})

if __name__=="__main__":
    app.run(debug=True)

