from flask import Flask,request,render_template
import pandas
import numpy as np
import pickle


breast_cancer=pickle.load(open("breast_cancer.pkl",'rb'))
 
app=Flask(__name__)
 
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict" ,methods=['POST'])
def predict():
    feature=request.form['feature']
    feature_list=feature.split(',')
    np_feature=np.asarray(feature_list,dtype=np.float32)
    pred=breast_cancer.predict(np_feature.reshape(1,-1))

    # output = ["Cancerous" if pred[0] == 1 else "Not Cancerous" if pred[0] == 0 else "Error"]


    output=["cancerous" if pred[0]==1 else "Not Cancerous"]

    return render_template("index.html", message=output)


if __name__=="__main__":
    app.run(debug=True)