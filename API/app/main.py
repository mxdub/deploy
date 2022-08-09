# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
import pandas as pd
import shap
import pickle
import sklearn
import xgboost
import requests
import numpy as np

app = Flask(__name__)

# Read Scaler
with open('standard_scaler.pkl', 'rb') as inp:
    ss = pickle.load(inp)

# Read model
with open('xgb_model.pkl', 'rb') as inp:
    xgb = pickle.load(inp)

# Read explainer
with open('shap_xgb_explainer.pkl', 'rb') as inp:
    xgb_explainer = pickle.load(inp)

# Read data (X)
df = pd.read_csv('./us_data_subset.csv')
df_unscaled = df.copy()

# Get targets & ids as separate lists
target = df.pop('TARGET')
sk_id_curr = df.pop('SK_ID_CURR')

# Scaled df
df_ss = pd.DataFrame(ss.transform(df), columns = df.columns)
probs = xgb.predict_proba(df_ss)[:,1]

# Get fn, fp, tp, tn 
def fp_fn_tp_tn(pp_, y_):
    thresh = np.arange(0,1,0.025)
    
    output = []

    for t in thresh:
        yp_ = (pp_ > t).astype('int')
        cm = sklearn.metrics.confusion_matrix(y_, yp_)
        tn_ = cm[0,0]
        fp_ = cm[0,1]
        fn_ = cm[1,0]
        tp_ = cm[1,1]
        prec_ = tp_ / (tp_ + fp_)
        recall_ = tp_ / (tp_ + fn_)
        output.append([t, tn_, fp_, fn_, tp_, prec_, recall_])
    
    return pd.DataFrame(output, columns = ['threshold', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall'])

stats = fp_fn_tp_tn(probs, target)

# Define endpoints
@app.route('/')
def test():
    return """Welcome to the API for Credit Scoring ! <br/>
            <br/>
            Available Endpoints (through HTTP request):
            <ul>
                <li>/get_data/</li> : return unscaled dataset
                <li>/get_idx/</li> : return IDs list
                <li>/predict/</li> : return predictions
                <li>/get_stats/</li> : return model statistics (fn, fp, etc.)
                <li>/get_shaps/ (POST request)</li> : return Shapleys values for one user (using 'id' variable of POST request)
            </ul>

            <br/>
            Dashboard can be find <a href="https://credits-ocr-dashboard.herokuapp.com/">here</a>.
            """

@app.route('/get_data/')
def get_df():
    return df_unscaled.to_json()

@app.route('/get_idx/')
def get_idx():
    return sk_id_curr.to_json()

@app.route('/predict/')
def get_predictions():
    probs_dt = pd.DataFrame({ 'SK_ID_CURR' : sk_id_curr,
                           'probs' : probs })
    return probs_dt.to_json()

@app.route('/get_stats/')
def get_acc_stats():
    return stats.to_json()

@app.route('/get_shaps/', methods = ['POST'])
def get_shaps():

    try :
        id = int(request.form['id'])
    except ValueError:
        print("Needed key missing (id)")

    loc_idx = list(sk_id_curr).index(id)

    # Apply shapley explainer
    shap_values = xgb_explainer(df_ss.iloc[[loc_idx]])

    # Reformat to dict{base_(float32), values_(panda)[feature names, values, data]}
    shap_info = {'base_value': float(shap_values.base_values[0]),
                 'shap_data' : pd.DataFrame({ 'feature_names' : xgb.feature_names_in_.tolist(),
                                             'shap_values' : shap_values.values[0].tolist(),
                                             'data' : shap_values.data[0].tolist()}).sort_values('shap_values', key = abs, ascending = False).to_dict()}

    return jsonify(shap_info)

if __name__ == "__main__":
    app.run(debug=True)
    