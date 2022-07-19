# -*- coding: utf-8 -*-
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import os

df = requests.get('https://credits-ocr-flaskapi.herokuapp.com//get_data/').json()
df = pd.DataFrame(df)

predicts = requests.post('https://credits-ocr-flaskapi.herokuapp.com//predict/', data={'id': 'all'}).json()
predicts = pd.Series(predicts, name = 'predictions')


GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 5000)

app = Dash(
    __name__,
    #meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

app.title = "Testing DASh ! "

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

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
                html.Div(
                    [
                        html.A(
                            html.Button("SOURCE CODE", className="link-button"),
                            href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-wind-streaming",
                        ),
                        html.A(
                            html.Button("ENTERPRISE DEMO", className="link-button"),
                            href="https://plotly.com/get-demo/",
                        ),
                        html.A(
                            html.Img(
                                src=app.get_asset_url("dash-new-logo.png"),
                                className="app__menu__img",
                            ),
                            href="https://plotly.com/dash/",
                        ),
                    ],
                    className="app__header__logo",
                ),
            ],
            className="app__header",
        ),
        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6("WIND SPEED (MPH)", className="graph__title"), html.H6("WIND SPEED (MPH)", className="graph__title")]
                                ),
                                html.Div([html.H6("WIND SPEED (MPH)", className="graph__title")]
                                )
                            ],
                        ),
                        dcc.Graph(
                            id="wind-speed",
                            # figure=dict(
                            #     layout=dict(
                            #         plot_bgcolor=app_color["graph_bg"],
                            #         paper_bgcolor=app_color["graph_bg"],
                            #     )
                            # ),
                        ),
                        # dcc.Interval(
                        #     id="wind-speed-update",
                        #     interval=int(GRAPH_INTERVAL),
                        #     n_intervals=0,
                        # ),
                    ],
                    className="two-thirds column wind__speed__container",
                ),
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
                                        )
                                    ]
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(list(df.index),list(df.index)[-1],id='id_client')
                                        # dcc.Slider(
                                        #     id="bin-slider",
                                        #     min=1,
                                        #     max=60,
                                        #     step=1,
                                        #     value=20,
                                        #     updatemode="drag",
                                        #     marks={
                                        #         20: {"label": "20"},
                                        #         40: {"label": "40"},
                                        #         60: {"label": "60"},
                                        #     },
                                        # )
                                    ],
                                    className="dropdown",
                                ),
                                # html.Div(
                                #     [
                                #         dcc.Checklist(
                                #             id="bin-auto",
                                #             options=[
                                #                 {"label": "Auto", "value": "Auto"}
                                #             ],
                                #             value=["Auto"],
                                #             inputClassName="auto__checkbox",
                                #             labelClassName="auto__label",
                                #         ),
                                #         html.P(
                                #             "# of Bins: Auto",
                                #             id="bin-size",
                                #             className="auto__p",
                                #         ),
                                #     ],
                                #     className="auto__container",
                                # ),
                                dcc.Graph(
                                    id="wind-histogram",
                                    # figure=dict(
                                    #     layout=dict(
                                    #         plot_bgcolor=app_color["graph_bg"],
                                    #         paper_bgcolor=app_color["graph_bg"],
                                    #     )
                                    # ),
                                ),
                            ],
                            className="graph__container first",
                        ),
                        # wind direction
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "WIND DIRECTION", className="graph__title"
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="wind-direction",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container second",
                        ),
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
    Output("wind-speed", "figure"), [Input("id_client", "value")]
)
def plot_data(id):
    """
    blabla
    """

    mask = np.zeros(df.shape[0])
    mask[int(id)] = 1
    fig = px.scatter(df, x = 'x1', y = 'y', hover_name=df.index ,
        color = mask.astype('str'), 
        size = mask + 1,
        color_discrete_map={"0.0": "blue","1.0": "red",},
        )

    fig.update_layout(    
        dict(
            plot_bgcolor=app_color["graph_bg"],
            paper_bgcolor=app_color["graph_bg"],
            font={"color": "#fff"},
            # height=700,
            showlegend=False,
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
                "nticks": 6,
            },
        )
    )

    return fig


@app.callback(
    Output("wind-histogram", "figure"), [Input("id_client", "value")]
)
def hist_probs(id):
    """
    blabla
    """

    mask = np.zeros(df.shape[0])
    mask[int(id)] = 1
    fig = px.histogram(predicts, x = 'predictions', histnorm = 'percent', nbins = 20)
    
    fig.add_vline(x=predicts[int(id)], line_dash = 'dash', line_color = 'white')

    fig.update_layout(    
        dict(
            plot_bgcolor=app_color["graph_bg"],
            paper_bgcolor=app_color["graph_bg"],
            font={"color": "#fff"},
            # height=700,
            xaxis={
                "range": [0, 1],
                "showline": True,
                "zeroline": False,
                "fixedrange": True,
                # "tickvals": [0, 50, 100, 150, 200],
                # "ticktext": ["200", "150", "100", "50", "0"],
                "title": "Success prob.",
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