#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 23:32:41 2020

@author: vesis
"""
# Import necessary libraries
# Jee jee
# Jee jee
# Jee jee
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os
%matplotlib inline

from plotly import  __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Working directory
os.chdir('/home/vesis/Documents/Python')

print(__version__)

init_notebook_mode(connected=True)

portfolio_df = pd.read_excel('Sample stocks acquisition dates_costs.xlsx')#, sheetname='Sample')
portfolio_df.head(10)

# Define the date variables
start_sp = datetime.datetime(2013, 1, 1)                  
end_sp = datetime.datetime(2018, 3, 9)                  
end_of_last_year = datetime.datetime(2017, 12, 29)
start_stocks = datetime.datetime(2013, 1, 1)                  
end_stocks = datetime.datetime(2018, 3, 7)      # Muutettu päivää aiemmaksi

# Pulling the SP 500 data from Yahoo! Finance
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

sp500 = pdr.get_data_yahoo('^GSPC', start_sp, end_sp)
sp500.head()
sp500.tail()

# SP 500 data including only the closing value column
sp_500_adj_close = sp500[['Adj Close']].reset_index()
sp_500_adj_close.tail()

# Last year's value to be used in the YTD calculation
sp_500_adj_close_start = sp_500_adj_close[sp_500_adj_close['Date'] == end_of_last_year]

# Pulling the tickers' data from Yahoo! Finance
tickers = portfolio_df['Ticker'].unique()

def get(tickers, startdate, enddate):
    def data(ticker):
        return(pdr.get_data_yahoo(ticker, start = startdate, end = enddate))
    datas = map(data, tickers)
    return pd.concat(datas, keys = tickers, names = ['Ticker', 'Date'])

all_data = get(tickers, start_stocks, end_stocks)

all_data.head()

# Stocks' data including only the closing value column
adj_close = all_data[['Adj Close']].reset_index()
adj_close.head()
adj_close.tail()

adj_close_start = adj_close[adj_close['Date'] == end_of_last_year]
adj_close_start.head(10)

adj_close_latest = adj_close[adj_close['Date'] == end_stocks]
adj_close_latest.head(10)

# Set Tickers as indexes
adj_close_latest.set_index(['Ticker'], inplace=True)
adj_close_latest.head()

portfolio_df.set_index(['Ticker'], inplace=True)
portfolio_df.head()

# Merge portfolio_df with with adj_close 
merged_portfolio = pd.merge(portfolio_df, adj_close_latest, left_index=True, right_index=True)
merged_portfolio.head(10)

# Create a 'ticker return' column stating the return for each ticker
merged_portfolio['ticker return'] = merged_portfolio['Adj Close'] / merged_portfolio['Unit Cost']-1
merged_portfolio

# Resetting the index
merged_portfolio.reset_index(inplace=True)

# Merge merged_portfolio with SP 500 adj_close values based on the acquisition date
merged_portfolio_sp = pd.merge(merged_portfolio, sp_500_adj_close, left_on='Acquisition Date', right_on='Date')    
merged_portfolio_sp.head()

# Delete double Date column and rename columns
del merged_portfolio_sp['Date_y']
merged_portfolio_sp.rename(columns={'Date_x':'Latest Date','Adj Close_x':'Ticker Adj Close','Adj Close_y':'SP 500 Initial Close'}, inplace=True)
merged_portfolio_sp.head()

# SP 500 eq shares
merged_portfolio_sp['Equiv SP Shares'] = merged_portfolio_sp[' '] / merged_portfolio_sp['SP 500 Initial Close']
merged_portfolio_sp.head()

# Merge merged_portfolio_sp with SP 500 adj_close values based on the end date
merged_portfolio_sp_latest = pd.merge(merged_portfolio_sp, sp_500_adj_close, left_on='Latest Date', right_on='Date')
merged_portfolio_sp_latest.head()        

# Delete double Date column and rename columns
del merged_portfolio_sp_latest['Date']
merged_portfolio_sp_latest.rename(columns={'Adj Close':'SP 500 Latest Close'}, inplace=True)
merged_portfolio_sp_latest.head()

# Define a bunch of columns
# Percent return of SP from acquisition date of position through latest trading day.
merged_portfolio_sp_latest['SP Return'] = merged_portfolio_sp_latest['SP 500 Latest Close'] / merged_portfolio_sp_latest['SP 500 Initial Close'] - 1

# This is a new column which takes the tickers return and subtracts the sp 500 equivalent range return.
merged_portfolio_sp_latest['Abs. Return Compare'] = merged_portfolio_sp_latest['ticker return'] - merged_portfolio_sp_latest['SP Return']

# This is a new column where we calculate the ticker's share value by multiplying the original quantity by the latest close.
merged_portfolio_sp_latest['Ticker Share Value'] = merged_portfolio_sp_latest['Quantity'] * merged_portfolio_sp_latest['Ticker Adj Close']

# We calculate the equivalent SP 500 Value if we take the original SP shares * the latest SP 500 share price.
merged_portfolio_sp_latest['SP 500 Value'] = merged_portfolio_sp_latest['Equiv SP shares'] * merged_portfolio_sp_latest['SP 500 Latest Close']

# This is a new column where we take the current market value for the shares and subtract the SP 500 value.
merged_portfolio_sp_latest['Abs Value Compare'] = merged_portfolio_sp_latest['Ticker Share Value'] - merged_portfolio_sp_latest['SP 500 Value']

# This column calculates profit / loss for stock position.
merged_portfolio_sp_latest['Stock Gain / (Loss)'] = merged_portfolio_sp_latest['Ticker Share Value'] - merged_portfolio_sp_latest['Cost Basis']

# This column calculates profit / loss for SP 500.
merged_portfolio_sp_latest['SP 500 Gain / (Loss)'] = merged_portfolio_sp_latest['SP 500 Value'] - merged_portfolio_sp_latest['Cost Basis']
merged_portfolio_sp_latest.head()

# Merge merged_portfolio_sp_latest with adj_close_start to track YTD performance
merged_portfolio_sp_latest_YTD = pd.merge(merged_portfolio_sp_latest, adj_close_start, on = 'Ticker')
merged_portfolio_sp_latest_YTD.head()

# Delete double date and rename columns
del merged_portfolio_sp_latest_YTD['Date']
merged_portfolio_sp_latest_YTD.rename(columns={'Adj Close':'Ticker Start Year Close'}, inplace=True)
merged_portfolio_sp_latest_YTD.head()

# Merge merged_portfolio_sp_latest with adj_close_start to track SP 500 YTD performance
merged_portfolio_sp_latest_YTD_sp = pd.merge(merged_portfolio_sp_latest_YTD, sp_500_adj_close_start, left_on = 'Start of Year', right_on = 'Date')
merged_portfolio_sp_latest_YTD_sp.head()

# Delete double date and rename columns
del merged_portfolio_sp_latest_YTD_sp['Date']
merged_portfolio_sp_latest_YTD_sp.rename(columns={'Adj Close':'SP Start Year Close'}, inplace=True)
merged_portfolio_sp_latest_YTD_sp.head()

# YTD returns
merged_portfolio_sp_latest_YTD_sp['Share YTD'] = merged_portfolio_sp_latest_YTD_sp['Ticker Adj Close'] / merged_portfolio_sp_latest_YTD_sp['Ticker Start Year Close']-1
merged_portfolio_sp_latest_YTD_sp['SP 500 YTD'] = merged_portfolio_sp_latest_YTD_sp['SP 500 Latest Close'] / merged_portfolio_sp_latest_YTD_sp['SP Start Year Close']-1
merged_portfolio_sp_latest_YTD_sp.head()

# Sort by Ticker
merged_portfolio_sp_latest_YTD_sp = merged_portfolio_sp_latest_YTD_sp.sort_values(by = 'Ticker', ascending=True)
merged_portfolio_sp_latest_YTD_sp 

# CumSum of original investment
merged_portfolio_sp_latest_YTD_sp['Cum Invst'] = merged_portfolio_sp_latest_YTD_sp['Cost Basis'].cumsum()
# CumSum of Ticker share value
merged_portfolio_sp_latest_YTD_sp['Cum Ticker Returns'] = merged_portfolio_sp_latest_YTD_sp['Ticker Share Value'].cumsum()
# CumSum of SP share value
merged_portfolio_sp_latest_YTD_sp['Cum SP Returns']  = merged_portfolio_sp_latest_YTD_sp['SP 500 Value'].cumsum()
# Cum CoC multiple return for stock investments
merged_portfolio_sp_latest_YTD_sp['Cum Ticker ROI Mult'] = merged_portfolio_sp_latest_YTD_sp['Cum Ticker Returns']/merged_portfolio_sp_latest_YTD_sp['Cum Invst']

merged_portfolio_sp_latest_YTD_sp.head()

# A Little recap - the starting tables
adj_close.head()
portfolio_df.head()

# Join adj_close with portfolio_df
portfolio_df.reset_index(inplace=True)
adj_close_acq_date = pd.merge(adj_close, portfolio_df, on = 'Ticker')
adj_close_acq_date.head()

# Delete some columns and order
del adj_close_acq_date['Quantity']
del adj_close_acq_date['Unit Cost']
del adj_close_acq_date['Cost Basis']
del adj_close_acq_date['Start of Year']
adj_close_acq_date.sort_values(by=['Ticker', 'Acquisition Date', 'Date'], ascending=[True, True, True], inplace=True)

# Calculate the difference between Date and Acquisition Date
adj_close_acq_date['Date Delta'] = adj_close_acq_date['Date']-adj_close_acq_date['Acquisition Date']
adj_close_acq_date['Date Delta'] = adj_close_acq_date[['Date Delta']].apply(pd.to_numeric)
adj_close_acq_date.head()

# Save only observations that take place after the Acquisition date
adj_close_acq_date_modified = adj_close_acq_date[adj_close_acq_date['Date Delta'] >= 0]
adj_close_acq_date_modified.head()

# Pivot table
adj_close_pivot = adj_close_acq_date_modified.pivot_table(index=['Ticker', 'Acquisition Date'], values='Adj Close', aggfunc=np.max) 
adj_close_pivot.reset_index(inplace = True)
adj_close_pivot

# Merge adj_close_pivot with adj_close to get the date of the Adj_close high
adj_close_pivot_merged = pd.merge(adj_close_pivot, adj_close, on = ['Ticker', 'Adj Close'])
adj_close_pivot_merged.head()

# Merge adj_close_pivot_merged with the master data frame
merged_portfolio_sp_latest_YTD_sp_closing_high = pd.merge(merged_portfolio_sp_latest_YTD_sp, adj_close_pivot_merged, on = ['Ticker', 'Acquisition Date'])
merged_portfolio_sp_latest_YTD_sp_closing_high.rename(columns={'Adj Close':'Closing High Adj Close', 'Date':'Closing High Adj Close Date'}, inplace=True)
merged_portfolio_sp_latest_YTD_sp_closing_high['Pct off High'] = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker Adj Close']/merged_portfolio_sp_latest_YTD_sp_closing_high['Closing High Adj Close']-1

merged_portfolio_sp_latest_YTD_sp_closing_high

# Create plots using plotly
# Plot 1: YTD return vs SP 500 YTD
trace1 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp['Share YTD'][0:10],
    name = 'Ticker YTD')

trace2 = go.Scatter(
    x = merged_portfolio_sp_latest_YTD_sp['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp['SP 500 YTD'][0:10],
    name = 'SP500 YTD')
    
data = [trace1, trace2]

layout = go.Layout(title = 'YTD Return vs S&P 500 YTD'
    , barmode = 'group'
    , yaxis=dict(title='Returns', tickformat=".2%")
    , xaxis=dict(title='Ticker')
    , legend=dict(x=.8,y=1)
    )

fig = go.Figure(data=data, layout=layout)
plot(fig)

iplot(fig)

# Plot 2: Total return vs SP 500
trace1 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['ticker return'][0:10],
    name = 'Ticker Total Return')

trace2 = go.Scatter(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['SP Return'][0:10],
    name = 'SP500 Total Return')
    
data = [trace1, trace2]

layout = go.Layout(title = 'Total Return vs S&P 500'
    , barmode = 'group'
    , yaxis=dict(title='Returns', tickformat=".2%")
    , xaxis=dict(title='Ticker', tickformat=".2%")
    , legend=dict(x=.8,y=1)
    )

fig = go.Figure(data=data, layout=layout)
plot(fig)

# Plot 3: Cumulative return over time vs SP 500
trace1 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Stock Gain / (Loss)'][0:10],
    name = 'Ticker Total Return ($)')

trace2 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['SP 500 Gain / (Loss)'][0:10],
    name = 'SP 500 Total Return ($)')

trace3 = go.Scatter(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['ticker return'][0:10],
    name = 'Ticker Total Return %',
    yaxis='y2')

data = [trace1, trace2, trace3]

layout = go.Layout(title = 'Gain / (Loss) Total Return vs S&P 500'
    , barmode = 'group'
    , yaxis=dict(title='Gain / (Loss) ($)')
    , yaxis2=dict(title='Ticker Return', overlaying='y', side='right', tickformat=".2%")
    , xaxis=dict(title='Ticker')
    , legend=dict(x=.75,y=1)
    )

fig = go.Figure(data=data, layout=layout)
plot(fig)

# Plot 4: Cum investments over time and returns 
trace1 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum Invst'],
    # mode = 'lines+markers',
    name = 'Cum Invst')

trace2 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum SP Returns'],
    # mode = 'lines+markers',
    name = 'Cum SP500 Returns')

trace3 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum Ticker Returns'],
    # mode = 'lines+markers',
    name = 'Cum Ticker Returns')

trace4 = go.Scatter(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Cum Ticker ROI Mult'],
    # mode = 'lines+markers',
    name = 'Cum ROI  Mult'
    , yaxis='y2')


data = [trace1, trace2, trace3, trace4]

layout = go.Layout(title = 'Total Cumulative Investments Over Time'
    , barmode = 'group'
    , yaxis=dict(title='Returns')
    , xaxis=dict(title='Ticker')
    , legend=dict(x=.4,y=1)
    , yaxis2=dict(title='Cum ROI Mult', overlaying='y', side='right')               
    )

fig = go.Figure(data=data, layout=layout)
plot(fig)

# Plot 5: Current share price vs closing high since purchased
trace1 = go.Bar(
    x = merged_portfolio_sp_latest_YTD_sp_closing_high['Ticker'][0:10],
    y = merged_portfolio_sp_latest_YTD_sp_closing_high['Pct off High'][0:10],
    name = 'Pct off High')
    
data = [trace1]

layout = go.Layout(title = 'Adj Close % off of High'
    , barmode = 'group'
    , yaxis=dict(title='% Below Adj Close High', tickformat=".2%")
    , xaxis=dict(title='Ticker')
    , legend=dict(x=.8,y=1)
    )

fig = go.Figure(data=data, layout=layout)
plot(fig)