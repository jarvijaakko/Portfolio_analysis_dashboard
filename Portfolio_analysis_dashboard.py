# Elaborated based on the wonderful tutorial by Kevin Boller (https://towardsdatascience.com/python-for-finance-dash-by-plotly-ccf84045b8be)

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

# Define the external CSS stylesheet to be used
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(external_stylesheets = external_stylesheets)

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
				dcc.Markdown(''' --- '''),
                
				# Cumulative Returns Over Time section
				html.H1('Cumulative Returns per Position Over Time'),
					dcc.Graph(id='crot1',
                                        figure = {'data':[
                                                go.Bar(
    											x = data['Ticker'][0:20],
    											y = data['Stock Gain / (Loss)'][0:20],
    											name = 'Ticker Total Return (Local currency)'),
    											go.Bar(
											    x = data['Ticker'][0:20],
											    y = data['OMXH 25 Gain / (Loss)'][0:20],
											    name = 'OMXH 25 Total Return (Local currenry)'),
											    go.Scatter(
    											x = data['Ticker'][0:20],
											    y = data['ticker return'][0:20],
											    name = 'Ticker Total Return %',
											    yaxis='y2')
		                                        ],
                                        'layout':go.Layout(title='Gain / (Loss) and Total Return vs OMXH 25',
                                        					barmode='group', 
                                                            xaxis = {'title': 'Ticker'},
                                                            yaxis = {'title': 'Gain / (Loss) (Local currency)'},
                                                            yaxis2 = {'title':'Ticker Return', 'overlaying':'y', 'side':'right', 'tickformat':".1%"},
                                                            #legend = {'x':'0.75', 'y':'1.2'}
                                         )}, style={'width': '100%'}
                                        ),
				dcc.Markdown(''' --- '''),
                
                # Total Cumulative Investments Over Time section
				html.H1('Total Cumulative Investments by Portfolio Over Time'),
					dcc.Graph(id='tcot1',
                                        figure = {'data':[
                                                go.Scatter(
    											x = data['Ticker'],
    											y = data['Cum Invst'],
    											mode = 'lines+markers',
    											name = 'Cum Invst'),
    											go.Scatter(
											    x = data['Ticker'],
											    y = data['Cum Ticker Returns'],
											    mode = 'lines+markers',
											    name = 'Cum Ticker Returns'),
											    go.Scatter(
    											x = data['Ticker'],
											    y = data['Cum OMXH Returns'],
											    mode = 'lines+markers',	
											    name = 'Cum OMXH 25 Returns'
											    )
		                                        ],
                                        'layout':go.Layout(title='Cumulative Investment Returns',
                                        					barmode='group', 
                                                            xaxis = {'title': 'Ticker'},
                                                            yaxis = {'title': 'Returns'},
                                                            #legend = {'x':'1', 'y':'1'}
                                         )}, style={'width': '100%'}
                                        ),
					dcc.Graph(id='tcot2',
                                        figure = {'data':[
                                                go.Bar(
											    x = data['Ticker'],
											    y = data['Cum Invst'],
											    name = 'Cum Invst'),
    											go.Bar(
											    x = data['Ticker'],
											    y = data['Cum OMXH Returns'],
											    name = 'Cum OMXH 25 Returns'),
											    go.Bar(
											    x = data['Ticker'],
											    y = data['Cum Ticker Returns'],
											    name = 'Cum Ticker Returns'),
											    go.Scatter(
											    x = data['Ticker'],
											    y = data['Cum Ticker ROI Mult'],
											    name = 'Cum ROI Mult'
											    , yaxis='y2'
											    )
		                                        ],
                                        'layout':go.Layout(title='Total Cumulative Investments | ROI Multiple, Over Time',
                                        					barmode='group', 
                                                            xaxis = {'title': 'Ticker'},
                                                            yaxis = {'title': 'Returns'},
                                                            yaxis2 = {'title':'Cum ROI Mult', 'overlaying':'y', 'side':'right'},
                                                            #legend = {'x':'0.75', 'y':'1.2'}
                                         )}, style={'width': '100%'}
                                        ),
					dcc.Markdown(''' --- '''),
                    # Current Share Price versus Closing High Since Purchased
					html.H1('Current Share Price versus Closing High Since Purchased'),
					dcc.Graph(id='cvh1',
                                        figure = {'data':[
                                                go.Bar(
    											x = data['Ticker'][0:20],
    											y = data['Pct off High'][0:20],
    											name = 'Pct off High'),
    											go.Scatter(
    											x = data['Ticker'][0:20],
    											y = [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25],
    											mode='lines',
    											name='Trailing Stop Marker',
    											line = {'color':'red'}
    											)
		                                        ],
                                        'layout':go.Layout(title='Adj Close % off of High Since Purchased',
                                        					barmode='group', 
                                                            xaxis = {'title': 'Ticker'},
                                                            yaxis = {'title': '% Below High Since Purchased', 'tickformat':'.2%'},
                                                            #legend = {'x':'0.8', 'y':'1.2'}
                                         )}, style={'width': '100%'}
                                        ),
				dcc.Markdown(''' --- ''')
])

# Append the external CSS stylesheet
app.css.append_css({
	'external_url' : 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

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