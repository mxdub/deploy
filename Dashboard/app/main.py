# -*- coding: utf-8 -*-
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
# import os

path = 'https://credits-ocr-flaskapi.herokuapp.com/'
# path = 'http://127.0.0.1:5000/'

predicts = requests.get(path + 'predict/').json()
predicts = pd.DataFrame(predicts)

df_unscaled = requests.get(path + 'get_data/').json()
df_unscaled = pd.DataFrame(df_unscaled)

sk_id_curr = sorted(list(requests.get(path + 'get_idx/').json().values()))

stats = pd.DataFrame(requests.get(path + 'get_stats/').json())

# GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 5000)

app = Dash(
    __name__,
    #meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

app.title = "Credit Scoring"

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}
acceptance_colors = {'Accepted':'#7BCD7B', 'Rejected':'#E06B6B'}

min_v_feats, max_v_feats, step_feats = 0, 20, 1

def logit(x):
    return np.log(x/(1-x))
                                                
app.layout = html.Div(
    [
        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4("CREDIT SCORING", className="app__header__title"),
                        html.P(
                            "Visualiser vos données crédit !",
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
            ],
            className="app__header",
        ),
        html.Div(
            [
                dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', 
                    parent_className = 'custom-tabs-parents two-thirds column', 
                    className='custom-tabs-container',
                    children=[
                    dcc.Tab(label='Client infos. (importance locale)', className='custom-tab', children = [
                        # wind speed
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H6("Top # feat. to display", 
                                                    className="parameters__style"),
                                                html.Div([ dcc.Slider(id="n_features", 
                                                    min = min_v_feats, max = max_v_feats, value = 10, step = step_feats,
                                                    marks = {v:str(v) for v in range(min_v_feats, max_v_feats+1, step_feats) if v % 5 == 0 },
                                                    tooltip={"placement": "bottom", "always_visible": True}) ]),                                        
                                            ], className="parameters__choice__subbox"),
                                        # html.Div(
                                        #     [
                                        #         []
                                        #     ], className="parameters__choice__subbox" ),
                                    ], className="parameters__choice__box"
                                ),
                                dcc.Graph(
                                    id="local-importance",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [html.H6('Compare to others', className="graph__title__bis"), 
                                             html.Div([
                                                html.P('Loans status : '),
                                                dcc.RadioItems(['Payed', 'Default','All status'], 'All status', inline=True, id='loans_status',
                                                    labelStyle = {'padding-right':'20px'}),
                                                ], style={'display': 'flex', 'padding':'5px'})
                                            ], 
                                            className="subtitle-band"
                                        ),
                                        html.Div(
                                            [
                                            ], 
                                            id = "multiple_hist_div", 
                                            className="multiple_histo_container"
                                        )
                                    ]
                                ),
                            ],
                            className="column wind__speed__container",
                        ),
                    ]),
                    dcc.Tab(label='Gestion du risque', className='custom-tab', children = [
                        html.Div(
                            [   
                                html.Div(
                                        [   
                                            html.Div(
                                                [
                                                    html.H6("beta value (f-score)", 
                                                        className="parameters__style"),
                                                    html.Div([ dcc.Slider(id="beta_fscore", 
                                                        min = 0, max = 5, value = 2,
                                                        tooltip={"placement": "bottom", "always_visible": True}) ]),
                                                ], className="parameters__choice__subbox"),
                                            html.Div([
                                                    html.H6("Selected threshold", 
                                                        className="parameters__style"),
                                                    html.Div([ dcc.Slider(id="threshold_choice", 
                                                        min = 0, max = 1, value = 0.34,
                                                        updatemode="drag",
                                                        tooltip={"placement": "bottom", "always_visible": True}) ]),
                                                ], className="parameters__choice__subbox"),
                                        ], className="parameters__choice__box"
                                    ),

                                dcc.Graph(
                                    id="risk_gestion_graph"
                                    ),
                            ], className="column wind__speed__container")
                    ])
                ]),

                html.Div(
                    [
                        # histogram
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Customer ID",
                                            className="graph__title",
                                        ),
                                        html.Div([
                                            dcc.Dropdown(sk_id_curr, sk_id_curr[0], id='id_client',)
                                        ], className = "dropdown"),
                                    ],
                                    className="customer-id-choice",
                                ),
                                dcc.Graph(
                                    id="predict_distrib",
                                ),
                            ],
                            className="graph__container first",
                        ),
                        # wind direction
                        # html.Div(
                        #     [
                        #         html.Div(
                        #             [
                        #                 html.H6(
                        #                     "WIND DIRECTION", className="graph__title"
                        #                 )
                        #             ]
                        #         ),
                        #         dcc.Graph(
                        #             id="wind-direction",
                        #             figure=dict(
                        #                 layout=dict(
                        #                     plot_bgcolor=app_color["graph_bg"],
                        #                     paper_bgcolor=app_color["graph_bg"],
                        #                 )
                        #             ),
                        #         ),
                        #     ],
                        #     className="graph__container second",
                        # ),
                    ],
                    className="one-third column histogram__direction",
                ),
            ],
            className="app__content",
        ),
    ],
    className="app__container",
)

@app.callback(
    Output("local-importance", "figure"), 
    Input("id_client", "value"),
    Input("n_features","value"),
    Input("threshold_choice", "value")
)
def plot_local_importance(id, n_features, thresh):
    """
    blabla
    """

    ## START block
    ## Works here - but note the extracted features (names) have to be used elsewhere
    ## This piece of code will be reused in another callback, using Dcc.store() to store the intermediate results
    ## might be a good solution to avoid running this twice but as it doen't take a while, let it as it is for now.

    rqst = requests.post(path + 'get_shaps/', {'id':id}).json()
    feat_imp = pd.DataFrame(rqst['shap_data']).sort_values('shap_values', key=abs)
    feat_imp_summary = feat_imp.iloc[-n_features:,:].copy()
    feature_names_list = list(feat_imp_summary.feature_names)
    
    ## END block
    # Gather unscaled features values
    unscaled_features_dict = {k:list(v.values())[0] for (k,v) in df_unscaled[df_unscaled.SK_ID_CURR == id].to_dict().items() if k in feature_names_list}
    
    feat_imp_summary.feature_names = [ '{:.2f} - {}'.format(unscaled_features_dict[v], v) for v in feat_imp_summary.feature_names]

    new_row = pd.DataFrame({'data':None, 'feature_names':'others', 'shap_values':sum(feat_imp.iloc[0:-n_features:,:].shap_values)}, index=[0])
    feat_imp_summary = pd.concat([new_row, feat_imp_summary])

    shap_base = rqst['base_value']

    fig = go.Figure(go.Waterfall(
        orientation = "h", 
        y = feat_imp_summary.feature_names,
        x = feat_imp_summary.shap_values,
        base = shap_base,
        connector = {"mode":"between", "line":{"width":4, "color":"rgb(0, 0, 0)", "dash":"solid"}},
        decreasing = {"marker":{"color":"#7BCD7B"}},
        increasing = {"marker":{"color":"#E06B6B"}},
    ))

    fig.update_layout(    
        dict(
            title = 'Most important features',
            plot_bgcolor=app_color["graph_bg"],
            paper_bgcolor=app_color["graph_bg"],
            font={"color": "#fff"},
            height=500,
            showlegend=False,
            margin=dict(b=110, t=110),
            xaxis={
                # "range": [0, 1],
                "showline": True,
                "zeroline": False,
                "fixedrange": True,
                # "tickvals": [0, 50, 100, 150, 200],
                # "ticktext": ["200", "150", "100", "50", "0"],
                # "title": "Success prob.",
            },
            yaxis={
                # "range": [0,100,],
                "showgrid": True,
                "showline": True,
                # "fixedrange": True,
                "zeroline": False,
                "gridcolor": app_color["graph_line"],
                # "nticks": 6,
            },
        )
    )

    # Classification threshold
    fig.add_vline(x=logit(thresh),
        y1 = 1.05, 
        line_dash = 'dash', 
        line_color = 'white',
        yref='paper')

    fig.add_annotation(
        x=logit(thresh),
        y=1.0,
        text= "Threshold",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        yref='paper'
        # bgcolor=acceptance_colors[preds_2.color[focal_idx]],
    )

    # Baseline shap value
    fig.add_vline(x=shap_base,
        y0 = -.1, 
        line_dash = 'dash', 
        line_color = 'grey',
        yref='paper')

    fig.add_annotation(
        x=shap_base,
        y=-.25,
        text= "Mean score",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        yref='paper'
        # bgcolor=acceptance_colors[preds_2.color[focal_idx]],
    )

    # Customer score
    fig.add_vline(x=shap_base + sum(feat_imp_summary.shap_values),
        y0 = -.2, 
        line_dash = 'dash', 
        line_color = 'white',
        yref='paper')

    fig.add_annotation(
        x=shap_base + sum(feat_imp_summary.shap_values),
        y=-.4,
        text= "Customer score",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        yref='paper'
        # bgcolor=acceptance_colors[preds_2.color[focal_idx]],
    )
    return fig

@app.callback(
    Output("risk_gestion_graph", "figure"), 
    Input("beta_fscore", "value"),
    Input("threshold_choice", "value")
)
def risk_plot(beta, thresh):

    stats_tm = stats.copy()

    stats_tm['beta'] = (1+beta**2)*stats['tp'] / ((1+beta**2)*stats['tp'] + beta**2 * stats['fn'] + stats['fp'])
    stats_tm['fp'] = stats_tm['fp'] / (stats_tm['fp'] + stats_tm['tn'])
    stats_tm['fn'] = stats_tm['fn'] / (stats_tm['fn'] + stats_tm['tp'])
    stats_tm.drop(columns = ['tn', 'tp'], inplace = True)
    stats_tm.drop(columns = ['precision', 'recall'], inplace = True)

    stats_tm = pd.melt(stats_tm, id_vars='threshold')

    fig = px.scatter(stats_tm, 
                        x="threshold", 
                        y="value", 
                        color="variable",
                        title="Model statistics")

    fig.add_vline(x=thresh, line_dash = 'dash', line_color = 'white', line_width=3.5)

    newnames = {'fn' : 'False negatives', 
                'fp' : 'False positives',
                'beta' : 'F-Score'}
    
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    fig.update_layout(    
        dict(
            plot_bgcolor=app_color["graph_bg"],
            paper_bgcolor=app_color["graph_bg"],
            font={"color": "#fff"},
            height=400,
            legend_title="",
            xaxis={
                "range": [0, 1],
                "showline": True,
                "zeroline": False,
                "fixedrange": True,
                # "tickvals": [0, 50, 100, 150, 200],
                # "ticktext": ["200", "150", "100", "50", "0"],
                "title": "Threshold",
            },
            yaxis={
                # "range": [0,100,],
                "showgrid": True,
                "showline": True,
                # "fixedrange": True,
                "zeroline": False,
                "gridcolor": app_color["graph_line"],
                "nticks": 6,
                "title":'',
            },
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        )
    )

    return fig

@app.callback(
    Output("multiple_hist_div", "children"),
    Input("id_client", "value"),
    Input("n_features","value"),
    Input("loans_status", "value")
)
def populate_hist_div(id, n_features, loan_status):

    rqst = requests.post(path + 'get_shaps/', {'id':id}).json()
    feat_imp = pd.DataFrame(rqst['shap_data']).sort_values('shap_values', key=abs)
    feat_imp_summary = feat_imp.iloc[-n_features:,:].copy()
    feature_names_list = list(feat_imp_summary.feature_names)
    unscaled_features_dict = {k:list(v.values())[0] for (k,v) in df_unscaled[df_unscaled.SK_ID_CURR == id].to_dict().items() if k in feature_names_list}

    if loan_status == 'Payed':
        filt = [0]
    elif loan_status == 'Default':
        filt = [1]
    else:
        filt = [0, 1]

    boxes_hist = []
    for feat in reversed(feature_names_list):
        subset = df_unscaled[ ['SK_ID_CURR', feat, 'TARGET'] ]

        focal_value = subset[subset.SK_ID_CURR == id]
        focal_value = focal_value[feat][0]

        subset = subset[ [True if x in filt else False for x in subset.TARGET] ]

        fig = px.histogram(subset, x = feat, color="TARGET", nbins = 50, histnorm = 'percent', opacity = 0.5, 
                           barmode="overlay", color_discrete_map = {0:'#7BCD7B',1:'#E06B6B'})

        if not np.isnan(focal_value):
            fig.add_vline(x=focal_value, line_dash = 'dash', line_color = 'white')

        fig.update_layout(    
            dict(
                title = feat,
                plot_bgcolor=app_color["graph_bg"],
                paper_bgcolor=app_color["graph_bg"],
                font={"color": "#fff"},
                font_size=10,
                height=300,
                width=400,
                showlegend=False,
                xaxis={
                    "showline": True,
                    "zeroline": False,
                    "fixedrange": True,
                    # "title": feat,
                },
                xaxis_title=None,
                yaxis={
                    "showgrid": True,
                    "showline": True,
                    "zeroline": False,
                    "gridcolor": app_color["graph_line"],
                    # "nticks": 6,
                },
            )
        )
        boxes_hist.append(dcc.Graph(figure = fig))
    return boxes_hist

@app.callback(
    Output("predict_distrib", "figure"), 
    Input("id_client", "value"),
    Input("threshold_choice","value")
)
def hist_probs(id, threshold):
    """
    Return histogram of predicted values (in logit scale) for all customers. 
    Colors represent acceptance status based on predicted probs from the model, and according to the threshold define in "Gestion du risque" tab.
    The white dashed line, represents predicted value for the choosen customer. 
    """

    try:
        focal_idx = list(predicts.SK_ID_CURR).index(id)
    except ValueError:
        print("Not a valid index !")

    mask = np.zeros(len(predicts.SK_ID_CURR))
    mask[focal_idx] = 1


    preds_2 = predicts.copy()

    # preds_2.probs = [logit(x) for x in predicts.probs]
    preds_2.probs = [(x) for x in predicts.probs]
    
    # preds_2['color'] = preds_2.probs < logit(threshold)
    preds_2['color'] = preds_2.probs < (threshold)

    preds_2.color = ['Accepted' if x else 'Rejected' for x in preds_2.color]

    # fig = px.histogram(preds_2, x = 'probs', color="color", histnorm = 'percent', nbins = 50)
    fig = px.histogram(preds_2, x = 'probs', color="color", nbins = 100, color_discrete_map = acceptance_colors)

    
    fig.add_vline(x=preds_2.probs[focal_idx], line_dash = 'dash', line_color = 'white')

    fig.add_annotation(
        x=preds_2.probs[focal_idx],
        y=1500,
        text=preds_2.color[focal_idx] + ' ({:.0f}%) '.format(100*predicts.probs[focal_idx]),
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor=acceptance_colors[preds_2.color[focal_idx]],
    )

    fig.update_layout(    
        dict(
            plot_bgcolor=app_color["graph_bg"],
            paper_bgcolor=app_color["graph_bg"],
            font={"color": "#fff"},
            height=300,
            legend_title="",
            xaxis={
                # "range": [-5, 3],
                "range": [0, 1],
                "showline": True,
                "zeroline": False,
                "fixedrange": True,
                # "tickvals": [0, 50, 100, 150, 200],
                # "ticktext": ["200", "150", "100", "50", "0"],
                # "title": "Score (logit scale)",
                "title": "Default prob.",
            },
            yaxis={
                # "range": [0,100,],
                "showgrid": True,
                "showline": True,
                # "fixedrange": True,
                "zeroline": False,
                "gridcolor": app_color["graph_line"],
                "nticks": 6,
            },
        )
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)