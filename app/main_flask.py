# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
import pandas as pd
import pickle
import sklearn
import requests

app = Flask(__name__)

df = pd.read_csv('./dataset.csv')
with open('lm_model.pkl', 'rb') as inp:
    lm = pickle.load(inp)

@app.route('/')
def test():
    return "Works !"

@app.route('/get_data/')
def get_df():
    return df.to_json()

@app.route('/predict/', methods=['POST'])
def get_predictions():
    data = request.form.get('id')
    if data == 'all':
        return pd.Series(lm.predict_proba(df.iloc[:,1:])[:,1], name = 'preds').to_json()

    # lm.predict()
    return 'df'

if __name__ == "__main__":
    app.run(debug=True)