import pickle
import webbrowser
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd

app=Flask(__name__)

#Load the model
regmodel=pickle.load(open('regmodel.pkl', 'rb'))
scalar=pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    input_data = np.array(list(data.values())).reshape(1,-1)
    new_data = scalar.transform(input_data)
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The House price prediction is {}".format(output))

if __name__=='__main__':
    # Automatically open the link in the default browser
    webbrowser.open_new("http://127.0.0.1:5001")
    app.run(debug=True, use_reloader=False, port=5001)