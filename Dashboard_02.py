#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:57:43 2020

@author: vesis
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
#from urllib.parse import urlencode
import pandas_datareader.data as web
import plotly.graph_objs as go
from datetime import datetime
import pandas as pd
import numpy as np
import os

# Working directory
os.chdir('/home/vesis/Documents/Python/Portfolio_analysis')

app = dash.Dash()

tickers = pd.read_csv('tickers.csv')
tickers.set_index('Ticker', inplace=True)

data = pd.read_csv('analyzed_portfolio.csv')

options = []

for tic in tickers.index:
	#{'label': 'user sees', 'value': 'script sees'}
	mydict = {}
	mydict['label'] = tic #Apple Co. AAPL
	mydict['value'] = tic
	options.append(mydict)


app.layout = html.Div([
				html.H1('Portfolio Dashboard'),
				dcc.Markdown(''' --- '''), 
				html.H1('Relative Returns Comparison'),
				html.Div([html.H3('Enter a stock symbol:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol',
						  options = options,
						  value = ['OMXH25'], 
						  multi = True
						  # style={'fontSize': 24, 'width': 75}
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
				html.Div([html.H3('Enter start / end date:'),
					dcc.DatePickerRange(id='my_date_picker',
										min_date_allowed = datetime(2015,1,1),
										max_date_allowed = datetime.today(),
										start_date = datetime(2018, 1, 1),
										end_date = datetime.today()
					)

				], style={'display':'inline-block'}), 
				html.Div([
					html.Button(id='submit-button',
								n_clicks = 0,
								children = 'Submit',
								style = {'fontSize': 24, 'marginLeft': '30px'}

					)

				], style={'display': 'inline-block'}),
				 
				dcc.Graph(id='my_graph',
							figure={'data':[
								{'x':[1,2], 'y':[3,1]}

							], 'layout':go.Layout(title='Relative Stock Returns Comparison', 
                                                            yaxis = {'title':'Returns', 'tickformat':".2%"}
                                         )}
				),
				dcc.Markdown(''' --- '''),

				# YTD Returns versus S&P 500 section
				html.H1('YTD and Total Position Returns versus OMXH 25'),
				 dcc.Graph(id='ytd1',
                                        figure = {'data':[
                                                go.Bar(
    											x = data['Ticker'][0:20],
    											y = data['Share YTD'][0:20],
    											name = 'Ticker YTD'),
    											go.Scatter(
											    x = data['Ticker'][0:20],
											    y = data['OMXH 25 YTD'][0:20],
											    name = 'OMXH 25 YTD')
                                                ],
                                        'layout':go.Layout(title='YTD Return vs OMXH 25 YTD',
                                        					barmode='group', 
                                                            xaxis = {'title':'Ticker'},
                                                            yaxis = {'title':'Returns', 'tickformat':".2%"}
                                         )}, style={'width': '50%', 'display':'inline-block'}
                                        ),
				# dcc.Markdown(''' --- '''),

				# Total Return Charts section
				# html.H1('Total Return Charts'),
					dcc.Graph(id='total1',
                                        figure = {'data':[
                                                go.Bar(
    											x = data['Ticker'][0:20],
    											y = data['ticker return'][0:20],
    											name = 'Ticker Total Return'),
    											go.Scatter(
											    x = data['Ticker'][0:20],
											    y = data['OMXH 25 Return'][0:20],
											    name = 'OMXH 25 Total Return')
                                                ],
                                        'layout':go.Layout(title='Total Return vs OMXH 25',
                                        					barmode='group', 
                                                            xaxis = {'title':'Ticker'},
                                                            yaxis = {'title':'Returns', 'tickformat':".2%"}
                                         )}, style={'width': '50%', 'display':'inline-block'}
                                        ),
				dcc.Markdown(''' --- ''')
])

@app.callback(Output('my_graph', 'figure'),
				[Input('submit-button', 'n_clicks')],
				[State('my_ticker_symbol', 'value'),
					  State('my_date_picker', 'start_date'),
					  State('my_date_picker', 'end_date')
				])
def update_graph(n_clicks, stock_ticker, start_date, end_date):
	start = datetime.strptime(start_date[:10], '%Y-%m-%d')
	end = datetime.strptime(end_date[:10], '%Y-%m-%d')

	traces = []
	for tic in stock_ticker:
		df = web.DataReader(tic, 'iex', start, end)
		traces.append({'x':df.index, 'y':(df['close']/df['close'].iloc[0])-1, 'name': tic})
	
	fig = {
		'data': traces,
		'layout': {'title':stock_ticker}
	}
	return fig

if __name__ == '__main__':
    app.run_server()