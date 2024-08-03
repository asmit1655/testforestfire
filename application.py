import pickle
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
import sklearn as sk

application=Flask(__name__)

#import ridge regressor and standard pickel file
ridge_model=pickle.load(open("models/ridge.pkl","rb"))
scaler_model=pickle.load(open("models/scaler.pkl","rb"))


@application.route("/")
def index():
    return render_template("index.html")

@application.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get("Temperature"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))
        
        scaled_data=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(scaled_data)
        
        return render_template("home.html",results=result[0])
    else:
        return render_template("home.html")
if __name__=="__main__":
    application.run()
