from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__) # Entry point to all

#app=application

## Route for a home page

@application.route('/') # render to index.html to search in templates folder
def index():
    return render_template('index.html') # to go to home page

@application.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            age=float(request.form.get('age')),
            heart_disease=int(request.form.get('heart_disease')),
            hypertension=int(request.form.get('hypertension')),
            ever_married=request.form.get('ever_married'),
            work_type=request.form.get('work_type'),
            Residence_type=request.form.get('Residence_type'),
            avg_glucose_level=float(request.form.get('avg_glucose_level')),
            bmi=float(request.form.get('bmi')),
            smoking_status=request.form.get('smoking_status')


        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        if results[0] == 0.0:
            return render_template('home.html',results="NO")
        else:
            return render_template('home.html',results="YES")
    
        # return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    application.run(host="0.0.0.0",debug=True)        

