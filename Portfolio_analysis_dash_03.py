import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import numpy as np
import datetime
import os

os.chdir('/home/vesis/Documents/Python/Portfolio_analysis')

app = dash.Dash(__name__)

tickers_saved = pd.read_csv('tickers.csv')
tickers_saved.set_index('Ticker', inplace=True)

# Function for pulling the ticker data
def get(tickers, startdate, enddate):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))

    datas = map(data, tickers)
    return pd.concat(datas, keys=tickers, names=['Ticker', 'Date'])

data = pd.read_csv('analyzed_portfolio.csv')

####
# Manipulation for the plot 3: OMXH 25 vs Ticker price development comparisons
chart_tickers = data['Ticker'].unique()
chart_tickers = chart_tickers.tolist()
chart_tickers.append('^OMXH25')
chart_tickers = np.array(chart_tickers)
ticker_amount = len(chart_tickers)

chart_start = datetime.datetime(2020, 1, 12)
chart_end = datetime.datetime(2020, 12, 29)

chart_data = get(chart_tickers, chart_start, chart_end)
chart_data.head()
chart_data.tail()
chart_data_eval = chart_data[['Adj Close']]
chart_data_eval.reset_index(inplace=True)

chart_data_eval_pivot = pd.pivot_table(chart_data_eval, index='Date', columns='Ticker', values='Adj Close')
chart_data_eval_pivot.reset_index(inplace=True)

# Interpolate missing values with values around
chart_data_eval_pivot_02 = chart_data_eval_pivot
mask = chart_data_eval_pivot_02.iloc[:, 1:].interpolate(axis=0, limit_area='inside', limit=1, limit_direction='backward').isna()
chart_data_eval_pivot_02_data = chart_data_eval_pivot_02.iloc[:, 1:].interpolate(axis=0, limit_area='inside', limit=1).mask(mask)

# Index prices to 100 (start date value)
ind_constant = chart_data_eval_pivot_02_data.loc[0]
chart_data_eval_pivot_02_data = (chart_data_eval_pivot_02_data/ind_constant)
chart_data_eval_pivot_02_filled = pd.merge(chart_data_eval_pivot_02, chart_data_eval_pivot_02_data, right_index=True, left_index=True, suffixes=('_x',''))

# Remove double columns after merging
columns_to_del = list(range(1,ticker_amount+1))
chart_data_eval_pivot_02_filled = chart_data_eval_pivot_02_filled.drop(chart_data_eval_pivot_02_filled.columns[columns_to_del], 1)
####

# Plot 3 build
trace1 = go.Scatter(
    x=chart_data_eval_pivot_02_filled['Date'],
    y=chart_data_eval_pivot_02_filled['AAPL'],
    mode='lines',
    name='AAPL Price'
)

data_3 = [trace1]

layout = go.Layout(title='Share Price development'
                   , barmode='group'
                   , yaxis=dict(title='Price (indexed)')
                   , xaxis=dict(title='Ticker')
                   , legend=dict(x=1, y=1)
                   )

fig_3 = go.Figure(data=data_3, layout=layout)

for x in chart_tickers[1:]:
    fig_3.add_trace(go.Scatter(x=chart_data_eval_pivot_02_filled['Date'],
                             y=chart_data_eval_pivot_02_filled[x],
                             mode='lines',
                             name=x + ' Price'
                             ))

# Plot 4: Pie chart of sectors
pie_pivot = pd.pivot_table(data,
                           index = 'Sector',
                           values = 'Ticker Share Value',
                           aggfunc = 'sum')
pie_pivot.reset_index(inplace=True)

# Plot 5: Pie chart of countries
pie_pivot_2 = pd.pivot_table(merged_portfolio,
                           index = 'Country',
                           values = 'Ticker Share Value',
                           aggfunc = 'sum')
pie_pivot_2.reset_index(inplace=True)

options = []

app = dash.Dash()
app.layout = html.Div([
    html.Div([
        html.H2('Portfolio Analysis Dashboard')
    ], className='banner'),
    
    html.Div([
        html.Div([
            html.H3('Returns'),
            dcc.Graph(id='g1', figure={'data': [
                go.Bar(
                    x=data['Ticker'][0:10],
                    y=data['Stock Gain / Loss'][0:10],
                    name='Ticker Total Return (Local currency)'),
                go.Bar(
                    x=data['Ticker'][0:10],
                    y=data['OMXH25 Gain / Loss'][0:10],
                    name='OMXH 25 Total Return (Local currency)'),
                go.Scatter(
                    x=data['Ticker'][0:10],
                    y=data['ticker return'][0:10],
                    name='Ticker Total Return (in %)',
                    yaxis='y2'),
                go.Scatter(
                    x=data['Ticker'][0:10],
                    y=data['OMXH25 Return'][0:10],
                    name='OMXH 25 Total Return (in %)',
                    yaxis='y2')
            ],
            'layout' : go.Layout(title='Total Return vs OMXH 25', 
                barmode='group',
                yaxis=dict(title='Gain / Loss (Local currency)'),
                yaxis2=dict(title='Return (%)', overlaying='y', side='right', tickformat=".2%"),
                xaxis=dict(title='Ticker'),
                legend=dict(x=0.05, y=1))})
        ],className="six columns"),
        html.Div([
            html.H3('Investments'),
            dcc.Graph(id='g2', figure={'data': [
                go.Bar(
                    x=data['Ticker'],
                    y=data['Cum Investments'],
                    # mode = 'lines+markers',
                    name='Cum Investments'),
                go.Bar(
                    x=data['Ticker'],
                    y=data['Cum Ticker Returns'],
                    # mode = 'lines+markers',
                    name='Cum Ticker Returns'),
                go.Scatter(
                    x=data['Ticker'],
                    y=data['Cum Ticker ROI Mult'],
                    # mode = 'lines+markers',
                    name='Cum ROI  Mult'
                    , yaxis='y2')
            ],
            'layout' : go.Layout(title='Total Cumulative Investments Over Time'
               , barmode='group'
               , yaxis=dict(title='Returns')
               , xaxis=dict(title='Ticker')
               , legend=dict(x=0.05, y=1)
               , yaxis2=dict(title='Cum ROI Mult', overlaying='y', side='right')
           )})
        ],className="six columns"),
    ],className="row"),
    html.Div([
        html.Div([
            html.H3('Prices'),
            dcc.Graph(id='g3', figure=fig_3
            )
        ],className="six columns"),
        html.Div([
            html.H3('Sectors'),
            dcc.Graph(id='g4', figure={'data': [
                go.Pie(
                    labels=pie_pivot['Sector'],
                    values=pie_pivot['Ticker Share Value'],
                )
            ],
})
        ],className="four columns"),
        html.Div([
            html.H3('Countries'),
            dcc.Graph(id='g5', figure={'data': [
                go.Pie(
                    labels=pie_pivot_2['Country'],
                    values=pie_pivot_2['Ticker Share Value'],
                )
            ],
})
        ],className="two columns"),
    ],className = "row")
])

# app.css.append_css({
#     'external_url' : 'https://codepen.io/chriddyp/pen/bWLwgP.css'
# })

app.server.route("/static/<path>")

if __name__ == '__main__':
    app.run_server()