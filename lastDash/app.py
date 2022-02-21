# -*- coding: utf-8 -*-
# todo: time and spectrogram x axis alignment.
# todo: global main_df to dash.Store json format test
# todo: dash.callback_context implementation to differentiate triggers

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from os import listdir
import pandas as pd
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_reusable_components as drc
import utils
import base64
import io
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import uuid
from sklearn.preprocessing import StandardScaler
from scipy import signal
import plotly.express as px
import plotly.io as pio
import dash_uploader as du

pio.templates.default = "none"
from dash import callback_context
import ruptures as rpt
import json

import matplotlib.pyplot as plt

layout_main = {
    "autosize": True,
    "paper_bgcolor": "#272a31",
    "plot_bgcolor": "#272a31",
    "margin": go.Margin(l=40, b=40, t=26, r=10),

    "xaxis": {
        "color": "white",
        "gridcolor": "#43454a",
        "tickwidth": 1,
    },

    "yaxis": {
        "color": "white",
        "gridcolor": "#43454a",
        "tickwidth": 1,
    }
}



fig = go.Figure(data=[], layout=layout_main)
fig = make_subplots(rows=2, cols=1, shared_yaxes=False, shared_xaxes=False, print_grid=True)
fig.update_layout(layout_main)

app = dash.Dash(__name__)
app.title = "IHA Visualization Tool"


main_df = pd.DataFrame()
temp_df = pd.DataFrame()
dataframes_list = []

ALLOWED_TYPES = (
    "text"
)

analyzeFilePath = ""
relayout_holder = {}


def get_upload_component(id):
    return du.Upload(
        id=id,
        max_file_size=1800,  # 1800 Mb
        filetypes=['csv', 'zip'],
        upload_id=uuid.uuid1(),  # Unique session id
    )


def serve_layout():
    # App Layout
    return html.Div(
        id="root",
        children=[
            # Session ID
            html.Div(id="session-id"),
            # Main body
            html.Div(
                id="app-container",
                children=[
                    # Banner display
                    html.Div(
                        id="banner",
                        children=[
                            html.Img(
                                id="logo", src=app.get_asset_url("dash.png")
                            ),
                            html.H2("IHA Data App", id="title"),
                        ],
                        style={'padding-bottom': "1rem"}
                    ),
                    # dcc.Tabs(id='tabs-example', value='tab-1', children=[
                    #     dcc.Tab(label='Tab one', value='tab-1'),
                    #     dcc.Tab(label='Tab two', value='tab-2'),
                    # ]),
                    html.Div(
                        id="image",
                        children=[
                            drc.Card([
                                drc.CustomDropdown(
                                    id="file_dropdown_for_multiple_graphs",
                                    options=[
                                    ],
                                    searchable=False,
                                    placeholder="Select file",
                                ),

                                drc.CustomDropdown(
                                    id="sensor_dropdown_for_multiple_graphs",
                                    options=[
                                    ],
                                    searchable=False,
                                    placeholder="Select sensor",
                                )
                                ]
                            ),
                            # The Interactive Image Div contains the dcc Graph
                            # showing the image, as well as the hidden div storing
                            # the true image
                            html.Div(
                                id="div-interactive-image",
                                children=[
                                    dcc.Graph(
                                        id="interactive-image",
                                        figure=fig,
                                        config={"displayModeBar": True},
                                    ),
                                    # html.Div(
                                    #         dcc.Graph(
                                    #             id="subplot_data",
                                    #             figure=fig,
                                    #             config={"displayModeBar": True},
                                    #         )),
                                    html.Div(
                                        id="div-storage",
                                        children=utils.STORAGE_PLACEHOLDER,
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
            # Sidebar
            html.Div(
                id="sidebar",
                children=[
                    drc.Card(
                        [
                            dcc.Store(id='main_df', storage_type='session'),
                            dcc.Upload(
                                id="upload_main",
                                children=[
                                    "Drag and Drop or ",
                                    html.B(children="Select a Telemetry File"),
                                ],
                                # No CSS alternative here
                                style={
                                    "color": "darkgray",
                                    "width": "100%",
                                    "height": "50px",
                                    "lineHeight": "50px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "borderColor": "darkgray",
                                    "textAlign": "center",
                                    "padding": "2rem 0",
                                    "margin-bottom": "2rem",
                                },
                            ),
                            drc.CustomDropdown(
                                id="main_telemetry_column",
                                options=[
                                ],
                                searchable=False,
                                placeholder="Telemetry Column..",
                            ),
                            drc.NamedInlineRadioItems(
                                name="NFFT",
                                short="nfft",
                                options=[
                                    {"label": " 128", "value": "128"},
                                    {"label": " 256", "value": "256"},
                                    {"label": " 512", "value": "512"},
                                    {"label": " 1024", "value": "1024"},
                                    {"label": " 2048", "value": "2048"},
                                ],
                                val="128",
                            ),
                            drc.NamedInlineRadioItems(
                                name="window_size",
                                short="windowsize",
                                options=[
                                    {"label": " 128", "value": "128"},
                                    {"label": " 256", "value": "256"},
                                    {"label": " 512", "value": "512"},
                                    {"label": " 1024", "value": "1024"},
                                    {"label": " 2048", "value": "2048"},
                                ],
                                val="128",
                            ),
                            drc.NamedInlineRadioItems(
                                name="mode",
                                short="mode",
                                options=[
                                    {"label": " journey", "value": "journey"},
                                    {"label": " analyse", "value": "analyse"},
                                ],
                                val="analysec",
                            ),
                        ]
                    ),
                    html.Hr(children=[], id="Hrobject",
                            style={"borderColor": "#FCCD61", "margin-top": "50px", "margin-bottom": "50px"}),
                    html.Pre(id='relayout-data',
                             style={'display': 'None', 'border': 'thin lightgrey solid', 'overflow': 'scroll'}),
                    html.Pre(id='relayout-data2',
                             style={'display': 'None', 'border': 'thin lightgrey solid', 'overflow': 'scroll'}),
                    drc.Card(
                        [dcc.Input(id="localFolderInput", type="text",
                                   placeholder="Copy your file location to analyze dataset", debounce=True, size="50",
                                   style={
                                       "width": "100%",
                                       "height": "50px",
                                       "lineHeight": "50px",
                                       "borderWidth": "1px",
                                       "borderRadius": "5px",
                                       "borderColor": "darkgray",
                                       "borderStyle": "dashed",
                                       "textAlign": "center",
                                       "margin-bottom": "2rem",
                                       "background-color": "#31343a",
                                       'font-size': '15px',
                                       'color': '#a6a9af'
                                   }, ),
                         drc.CustomDropdown(
                             id="csv_files_list",
                             options=[
                                 # {"label": "sensor.csv", "value" : "sensor.csv"},
                                 # {"label": "sensor2.csv", "value": "sensor2.csv"}
                             ],
                             searchable=True,
                             placeholder="Csv File Selection..",
                             multi=True,
                             style={"margin-bottom": "15px"}
                         ),
                         drc.CustomDropdown(
                             id="csv_attributes_dict",
                             options=[
                             ],
                             searchable=True,
                             placeholder="Telemetry Column Selection..",
                             multi=True,
                             style={"margin-bottom": "15px"}
                         ),
                         html.Div(
                             id="div-enhancement-factor",
                             children=[
                                 f"Enhancement Factor:",
                                 html.Div(
                                     children=dcc.Slider(
                                         id="slider-enhancement-factor",
                                         min=0,
                                         max=2,
                                         step=0.1,
                                         value=1,
                                         updatemode="drag",
                                     )
                                 ),
                             ],
                         ),
                         html.Div(
                             id="button-group",
                             children=[
                                 html.Button(
                                     "Run Operation", id="button-run-operation"
                                     , n_clicks=0,
                                 ),
                                 html.Button("Undo", id="button-undo"),
                             ],
                         ),
                         ]
                    ),

                ],
            ),
        ],
    )


def chunkProcess(dataarr, modelType):
    algo = []
    result = []
    for data in dataarr:
        algo.append(rpt.Pelt(model="rbf").fit(data))

    for al in algo:
        result.append(al.predict(pen=10))
    return algo


def changePointDetector(data, modelType, variables):
    data = data.to_numpy().reshape(-1, 1).copy()
    data = np.array_split(data, 100)
    chunkProcess(data, modelType)
    algo = rpt.Pelt(model="l1").fit(data)
    result = algo.predict(pen=10)

    return result


def spectrogram_calculation(data, nfft, window_size, frequency=1):
    """

    :param data: data to be calculated
    :param nfft: number of fft samples
    :param window_size: fft window size
    :param frequency: frequency of data
    :return: graph_objects heatmap.
    """

    # fit transform will be applied and nan values will be interpolated.
    data = data.to_numpy()
    data = StandardScaler().fit_transform(data.reshape(-1, 1))
    ok = ~np.isnan(data)
    xp = ok.ravel().nonzero()[0]
    fp = data[~np.isnan(data)]
    x = np.isnan(data).ravel().nonzero()[0]
    data[np.isnan(data)] = np.interp(x, xp, fp)

    # Spectrogram windows_size.
    w = signal.blackman(window_size)

    freqs, bins, Pxx = signal.spectrogram(data.reshape(1, -1), frequency, window=w, nfft=nfft)

    return go.Heatmap(
        x=bins,
        y=freqs,
        z=10 * np.log10(Pxx.squeeze()),
        colorscale='Jet',
        showlegend=False,
        legendgroup=False
    )


def parse_data(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif "txt" or "tsv" in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df


def find_csv_filenames(path_to_dir, suffix=".csv"):
    print("find csv file names'e girdik")
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


# CALLBACKS INITIALIZATION


@app.callback(
    Output("csv_attributes_dict", "options"),
    [Input("csv_files_list", "value")],
)
def attribute_render(values):
    print("attributa girdik")
    if values is None:
        raise PreventUpdate

    global temp_df
    global main_df

    # read_csv , parse_data falanla buraya inputtaki csv'yi yüklicez

    global dataframes_list
    list_of_columns = []
    for val in values:
        print("klasör: ", analyzeFilePath)
        print("dosya: ", val)
        temp_df = pd.read_csv(analyzeFilePath + '\\' + val)
        print("temp_df'in şekli: ", temp_df.shape)
        # df_parsed = parse_data(temp_df, val)
        dataframes_list.append({"name": val, "dataframe": temp_df})
        attribute_list = pd.read_csv(analyzeFilePath + '\\' + val, index_col=0, nrows=0).columns.tolist()

        for col in attribute_list:
            list_of_columns.append({"label": val + " \\ " + col, "value": val + "\\" + col})
    return list_of_columns


@app.callback(
    Output("csv_files_list", "options"),
    [Input("localFolderInput", "value")],
)
def cb_render(value):
    list_of_column = []
    print("cb_render'a girdik")
    if value is None:
        raise PreventUpdate
    global analyzeFilePath
    analyzeFilePath = value
    csv_file_names = find_csv_filenames(value)
    print("filenames: ??", csv_file_names)
    for col in csv_file_names:
        list_of_column.append({"label": col, "value": col})
    return list_of_column

@app.callback(
    Output("file_dropdown_for_multiple_graphs", "options"),
    [Input("csv_files_list", "value")],
)
def file_dropdown_to_dropdown(files):
    if files is None:
        raise PreventUpdate

    list_of_selected_files = []
    for item in files:
        list_of_selected_files.append({"label": item, "value": item})
    return list_of_selected_files

@app.callback(
    Output("sensor_dropdown_for_multiple_graphs", "options"),
    [Input("csv_attributes_dict", "value"),
     Input("file_dropdown_for_multiple_graphs", "value")],
)
def sensor_dropdown_to_dropdown(sensors, file):
    print(file)
    if sensors is None:
        raise PreventUpdate
    if file is None:
        raise PreventUpdate

    list_sensors_to_display = []
    for item in sensors:
        if file in item:
            list_sensors_to_display.append({"label": item, "value": item})
    return list_sensors_to_display


@app.callback(Output('main_telemetry_column', 'options'),
              Input('upload_main', 'contents'),
              Input('upload_main', 'filename'),
              State('main_df', 'data'))
def load_data(list_of_contents, list_of_names, data):
    if list_of_contents is None:
        PreventUpdate

    print("load_data ya girdik")
    list_of_column = []
    global main_df
    if list_of_contents is not None:

        main_df = parse_data(list_of_contents, list_of_names)
        for col in main_df.columns:
            list_of_column.append({"label": col, "value": col})

    return list_of_column


@app.callback(Output('relayout-data', 'children'),
              [Input('interactive-image', 'relayoutData')])
def display_relayout_data(relayoutData):
    global relayout_holder
    relayout_holder = relayoutData
    return json.dumps(relayoutData, indent=2)

#
# @app.callback(Output('subplot_data', 'figure'),
#               Input('interactive-image', 'relayoutData'),
#               Input("csv_files_list", "value"),
#               Input('csv_attributes_dict', 'value'),
#               [Input("localFolderInput", "value")],
#
#               )
# def subplot_output(relayoutData, values, dictValue, localFolder):
#     list_of_val = []
#     list_of_index = []
#     list_of_column = []
#     # print("localFolder")
#     # print(localFolder)
#     # print("val")
#     # print(val)
#     # print("values")
#     # print(values)
#
#     # global analyzeFilePath
#     list_of_columns = []
#     for val in values:
#         attribute_list = pd.read_csv(localFolder + '\\' + val)
#
#     # print(attribute_list)
#
#     dictValue = str(dictValue).split('+')
#     dictValue2 = dictValue[1].split('\'')
#     selected_column = dictValue2[0]
#     all_chosen_column = attribute_list[selected_column]
#
#     # print(all_chosen_column)
#
#     index_data1 = int((relayoutData.get("xaxis.range[0]")))
#     index_data2 = int((relayoutData.get("xaxis.range[1]")))
#
#     # attributed_data = main_df[val]
#
#     for i in range(index_data1, index_data2):
#         list_of_val.append(all_chosen_column[i])
#         list_of_index.append(i)
#
#     ff = make_subplots(rows=2, cols=1, shared_yaxes=True, shared_xaxes=True, print_grid=True)
#     ff.append_trace(trace=go.Scatter(y=list_of_val, x=list_of_index), row=1, col=1)
#     ff.update_layout(layout_main)
#     ff.update_xaxes(color='white', gridcolor="#43454a", tickwidth=1)
#     ff.update_yaxes(color='white', gridcolor="#43454a", tickwidth=1, row=1, col=1, autorange=True)
#     ff.update_yaxes(color='white', gridcolor="#43454a", tickwidth=1, row=2, col=1, autorange=True)
#     ff.update_traces(showscale=False, selector=dict(type='heatmap'))
#
#     return ff
#
#
# @app.callback(Output('interactive-image', 'figure'),
#               Input('main_telemetry_column', 'value'),
#               Input('radio-windowsize', 'value'),
#               Input('radio-nfft', 'value'),
#               Input('radio-mode', 'value'),
#               Input("button-run-operation", "n_clicks")
#               )
# def show_main_plot(val, window_size, nfft, mode, clicks):
#     global main_df
#     # print(dictValue)
#     ctx = dash.callback_context
#     ctxId = ctx.triggered[0]['prop_id'].split('.')[0]
#     if ctxId == 'button-run-operation':
#         if val is None:
#             raise PreventUpdate
#     else:
#         if val is None:
#             raise PreventUpdate
#         attributed_data = main_df[val]
#
#         attributed_timestamp = main_df['timestamp']
#         if mode == 'journey':
#
#             ff = make_subplots(rows=2, cols=1, shared_yaxes=False, shared_xaxes=False, print_grid=True)
#
#             ff.append_trace(trace=go.Scatter(y=attributed_data, x=attributed_timestamp), row=1, col=1)
#             spectrace = spectrogram_calculation(attributed_data, int(nfft), int(window_size), 1000)
#             ff.append_trace(trace=spectrace, row=2, col=1)
#             ff.update_layout(layout_main)
#             ff.update_xaxes(color='white', gridcolor="#43454a", tickwidth=1)
#             ff.update_yaxes(color='white', gridcolor="#43454a", tickwidth=1, row=1, col=1, autorange=True)
#             ff.update_yaxes(color='white', gridcolor="#43454a", tickwidth=1, row=2, col=1, autorange=True)
#             ff.update_traces(showscale=False, selector=dict(type='heatmap'))
#             fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
#         else:
#             ff = make_subplots(rows=2, cols=1, shared_yaxes=False, shared_xaxes=True, print_grid=True)
#             ff.append_trace(trace=go.Scatter(y=attributed_data), row=1, col=1)
#             spectrace = spectrogram_calculation(attributed_data, int(nfft), int(window_size), 1000)
#             ff.append_trace(trace=spectrace, row=2, col=1)
#             ff.update_layout(layout_main)
#             ff.update_xaxes(color='white', gridcolor="#43454a", tickwidth=1)
#             ff.update_yaxes(color='white', gridcolor="#43454a", tickwidth=1, row=1, col=1, autorange=True)
#             ff.update_yaxes(color='white', gridcolor="#43454a", tickwidth=1, row=2, col=1, autorange=True)
#             ff.update_traces(showscale=False, selector=dict(type='heatmap'))
#             fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
#         # ff = go.Figure(data=[go.Scatter(x=attributed_timestamp, y=attributed_data)], layout=layout_main)
#         # a = changePointDetector(attributed_data,'rbf',2)
#     return ff


@app.callback(Output('interactive-image', 'figure'),
              Input('sensor_dropdown_for_multiple_graphs', 'value'),
              Input('radio-windowsize', 'value'),
              Input('radio-nfft', 'value'),
              Input('radio-mode', 'value'),
              Input("button-run-operation", "n_clicks"),
              Input("csv_attributes_dict", "value"))

def show_main_plot_alt(val, window_size, nfft, mode, clicks, dictValue):
    global main_df
    ctx = dash.callback_context
    ctxId = ctx.triggered[0]['prop_id'].split('.')[0]
    if ctxId == 'button-run-operation':
        if dictValue is None and val is None:
            raise PreventUpdate

    else:
        if val is None:
            raise PreventUpdate

        [fileName, columnName] = val.split('\\')
        print("filename: ", fileName)
        print("column name: ", columnName)
        global dataframes_list
        for df in dataframes_list:
            if df["name"] == fileName:
                print("624. satır çalışmış")
                current_df = df["dataframe"]
        print("current_df'in şekli: ", current_df.shape)
        attributed_data = current_df[columnName]
        # print(attributed_data[4])
        attributed_timestamp = current_df['timestamp']
        if mode == 'journey':

            ff = make_subplots(rows=2, cols=1, shared_yaxes=False, shared_xaxes=False, print_grid=True)
            ff.append_trace(trace=go.Scatter(y=attributed_data, x=attributed_timestamp), row=1, col=1)
            spectrace = spectrogram_calculation(attributed_data, int(nfft), int(window_size), 1000)
            ff.append_trace(trace=spectrace, row=2, col=1)
            ff.update_layout(layout_main)
            ff.update_xaxes(color='white', gridcolor="#43454a", tickwidth=1)
            ff.update_yaxes(color='white', gridcolor="#43454a", tickwidth=1, row=1, col=1, autorange=True)
            ff.update_yaxes(color='white', gridcolor="#43454a", tickwidth=1, row=2, col=1, autorange=True)
            ff.update_traces(showscale=False, selector=dict(type='heatmap'))
            fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
        else:
            ff = make_subplots(rows=2, cols=1, shared_yaxes=False, shared_xaxes=True, print_grid=True)
            ff.append_trace(trace=go.Scatter(y=attributed_data), row=1, col=1)
            spectrace = spectrogram_calculation(attributed_data, int(nfft), int(window_size), 1000)
            ff.append_trace(trace=spectrace, row=2, col=1)
            ff.update_layout(layout_main)
            ff.update_xaxes(color='white', gridcolor="#43454a", tickwidth=1)
            ff.update_yaxes(color='white', gridcolor="#43454a", tickwidth=1, row=1, col=1, autorange=True)
            ff.update_yaxes(color='white', gridcolor="#43454a", tickwidth=1, row=2, col=1, autorange=True)
            ff.update_traces(showscale=False, selector=dict(type='heatmap'))
            fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
        # ff = go.Figure(data=[go.Scatter(x=attributed_timestamp, y=attributed_data)], layout=layout_main)
        # a = changePointDetector(attributed_data,'rbf',2)
    return ff


#
# @app.callback(Output('tabs-example-content', 'children'),
#               Input('tabs-example', 'value'))
# def render_content(tab):
#     if tab == 'tab-1':
#         return html.Div([
#             html.H3('Tab content 1'),
#
#         ])
#     elif tab == 'tab-2':
#         return html.Div([
#             html.H3('Tab content 2')
#         ])

app.layout = serve_layout()

if __name__ == '__main__':
    app.run_server(debug=False)
