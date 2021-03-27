# Inspired by the wonderful tutorial by Kevin Boller:
# (https://towardsdatascience.com/python-for-finance-stock-portfolio-analyses-6da4c3e61054)

# Import necessary libraries 
import pandas as pd
desired_width = 180
pd.set_option('display.width', desired_width)
pd.options.display.max_columns = None
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

# Working directory
os.chdir('/home/vesis/Documents/Python/Portfolio_analysis')
init_notebook_mode(connected=True)

# Define the date variables
# Make a function considering different opening dates and the time difference(!!!)
start_omxh = datetime.datetime(2011, 12, 29)
end_omxh = datetime.datetime(2020, 12, 29)
end_of_last_year = datetime.datetime(2019, 12, 30)
start_stocks = datetime.datetime(2011, 12, 29)
end_stocks = datetime.datetime(2020, 12, 29)

# DATA PULL
# Pull OMXH25 data from Yahoo! Finance
# Form a subset containing only daily closing values
# Define last year's closing value for YTD calculation
omxh25 = pdr.get_data_yahoo('^OMXH25', start_omxh, end_omxh)
omxh25_adj_close = omxh25[['Adj Close']].reset_index()
omxh25_adj_close_start = omxh25_adj_close[omxh25_adj_close['Date'] == end_of_last_year]

# PORTFOLIO DATA PULL #
# Read in the analyzed portfolio and pull corresponding data from Yahoo! Finance
portfolio_df = pd.read_excel('Stocks_v3.xls')
tickers = portfolio_df['Ticker'].unique()

# Function for pulling the ticker data
def get(tickers, startdate, enddate):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))

    datas = map(data, tickers)
    return pd.concat(datas, keys=tickers, names=['Ticker', 'Date'])

# Pull portfolio data from Yahoo! Finance
# Form a subset containing only daily closing values
# Define last year's closíng value for YTD calculation
# Define the latest closing value
portfolio = get(tickers, start_stocks, end_stocks)
portfolio_adj_close = portfolio[['Adj Close']].reset_index()
portfolio_adj_close_start = portfolio_adj_close[portfolio_adj_close['Date'] == end_of_last_year]
portfolio_adj_close_latest = portfolio_adj_close[portfolio_adj_close['Date'] == portfolio_adj_close['Date'].max()]

# DATA PULL loppuu

# GENERAL CALCULATION
# Set Tickers as indexes
portfolio_adj_close_latest.set_index(['Ticker'], inplace=True)
portfolio_df.set_index(['Ticker'], inplace=True)

# Merge portfolio_df with portfolio_adj_close_latest
merged_portfolio = pd.merge(portfolio_df, portfolio_adj_close_latest, left_index=True, right_index=True)

# Create a 'ticker return' column stating the return for each ticker
merged_portfolio['ticker return'] = merged_portfolio['Adj Close'] / merged_portfolio['Unit Cost'] - 1

# Resetting the index
merged_portfolio.reset_index(inplace=True)

# Merge merged_portfolio with OMXH25 closing values based on the acquisition date
# Delete double Date column and rename columns
merged_portfolio = pd.merge(merged_portfolio, omxh25_adj_close, left_on='Acquisition Date', right_on='Date')

del merged_portfolio['Date_y']
merged_portfolio.rename(
    columns={'Date_x': 'Latest Date', 'Adj Close_x': 'Ticker Adj Close', 'Adj Close_y': 'OMXH25 Initial Close'},
    inplace=True)

# Create a 'OMXH25 equivalent shares' column stating the equvalent amount of OMXH25 for each ticker
merged_portfolio['Equiv OMXH Shares'] = merged_portfolio['Cost Basis'] / merged_portfolio[
    'OMXH25 Initial Close']

# Merge merged_portfolio with OMXH25 adj_close values based on the end date
merged_portfolio = pd.merge(merged_portfolio, omxh25_adj_close, left_on='Latest Date', right_on='Date')

# Delete double Date column and rename columns
del merged_portfolio['Date']
merged_portfolio.rename(columns={'Adj Close': 'OMXH25 Latest Close'}, inplace=True)

# Define a bunch of columns
# Percent return of OMXH from acquisition date of position through latest trading day
merged_portfolio['OMXH25 Return'] = merged_portfolio['OMXH25 Latest Close'] / \
                                    merged_portfolio['OMXH25 Initial Close'] - 1

# Percentage difference of returns between the ticker and OMXH25
merged_portfolio['Abs. Return Compare'] = merged_portfolio['ticker return'] - \
                                          merged_portfolio['OMXH25 Return']

# Ticker's current share value
merged_portfolio['Ticker Share Value'] = merged_portfolio['Quantity'] * \
                                         merged_portfolio['Ticker Adj Close']

# Equivalent OMXH (monetary) value at current moment
merged_portfolio['OMXH25 Value'] = merged_portfolio['Equiv OMXH Shares'] * \
                                   merged_portfolio['OMXH25 Latest Close']

# Difference between ticker and OMXH (monetary) value
merged_portfolio['Abs Value Compare'] = merged_portfolio['Ticker Share Value'] - \
                                        merged_portfolio['OMXH25 Value']

# Profit / loss for stock position
merged_portfolio['Stock Gain / Loss'] = merged_portfolio['Ticker Share Value'] - \
                                        merged_portfolio['Cost Basis']

# Profit / loss for OMXH
merged_portfolio['OMXH25 Gain / Loss'] = merged_portfolio['OMXH25 Value'] - \
                                         merged_portfolio['Cost Basis']

# GENERAL CALCULATION loppuu

# YTD calculation
# Merge merged_portfolio with portfolio_adj_close_start to track YTD performance
merged_portfolio = pd.merge(merged_portfolio, portfolio_adj_close_start, on='Ticker')

# Delete double date and rename columns
del merged_portfolio['Date']
merged_portfolio.rename(columns={'Adj Close': 'Ticker Start Year Close'}, inplace=True)

# Merge merged_portfolio with OMXH_adj_close_start to track OMXH YTD performance
merged_portfolio = pd.merge(merged_portfolio, omxh25_adj_close_start,
                            left_on='Start of Year', right_on='Date')

# Delete double date and rename columns
del merged_portfolio['Date']
merged_portfolio.rename(columns={'Adj Close': 'OMXH25 Start Year Close'}, inplace=True)

# YTD returns (in percentage)
merged_portfolio['Share YTD Return'] = merged_portfolio['Ticker Adj Close'] / \
                                       merged_portfolio['Ticker Start Year Close'] - 1

merged_portfolio['OMXH25 YTD Return'] = merged_portfolio['OMXH25 Latest Close'] / \
                                        merged_portfolio['OMXH25 Start Year Close'] - 1

# YTD loppuu

# CUMULATIVE SUMS
# Sort rows by Acquisition Date
merged_portfolio = merged_portfolio.sort_values(by='Acquisition Date', ascending=False)

# CumSum of original investments
merged_portfolio['Cum Investments'] = merged_portfolio['Cost Basis'].cumsum()

# CumSum of Ticker share value
merged_portfolio['Cum Ticker Returns'] = merged_portfolio['Ticker Share Value'].cumsum()

# CumSum of OMXH share value
merged_portfolio['Cum OMXH Returns'] = merged_portfolio['OMXH25 Value'].cumsum()

# Cum CoC multiple return for stock investments
merged_portfolio['Cum Ticker ROI Mult'] = merged_portfolio['Cum Ticker Returns'] / \
                                          merged_portfolio['Cum Investments']

# CUMULATIVE SUMS loppuu

# HIGHEST VALUE
# Find out the highest value and its date after acquisition for each Ticker
# Join adj_close with portfolio_df
portfolio_df.reset_index(inplace=True)
highest_value_merged = pd.merge(portfolio_adj_close, portfolio_df, on='Ticker')

# Delete some columns and order
del highest_value_merged['Quantity']
del highest_value_merged['Unit Cost']
del highest_value_merged['Cost Basis']
del highest_value_merged['Start of Year']
highest_value_merged.sort_values(by=['Ticker', 'Acquisition Date', 'Date'], ascending=[True, True, True], inplace=True)

# Calculate the difference between Date and Acquisition Date
# Save only observations that take place after the Acquisition date
highest_value_merged['Date Delta'] = highest_value_merged['Date'] - highest_value_merged['Acquisition Date']
highest_value_merged['Date Delta'] = highest_value_merged[['Date Delta']].apply(pd.to_numeric)
highest_value_merged = highest_value_merged[highest_value_merged['Date Delta'] >= 0]

# Pivot table
highest_value_pivot = highest_value_merged.pivot_table(index=['Ticker', 'Acquisition Date'], 
                                                       values='Adj Close', 
                                                       aggfunc=np.max)
highest_value_pivot.reset_index(inplace=True)

# Merge adj_close_pivot with adj_close to get the date of the Adj_close high
highest_value_pivot = pd.merge(highest_value_pivot, portfolio_adj_close, on=['Ticker', 'Adj Close'])

# Merge adj_close_pivot_merged with the master data frame
# Rename columns
# Define how many percentages the current ticker value differs from the ticker's highest value after acquisition
merged_portfolio = pd.merge(merged_portfolio, highest_value_pivot, on=['Ticker', 'Acquisition Date'])

merged_portfolio.rename(columns={'Adj Close': 'Closing High Adj Close', 'Date': 'Closing High Adj Close Date'}, inplace=True)

merged_portfolio['Pct off High'] = merged_portfolio['Ticker Adj Close'] / \
                                   merged_portfolio['Closing High Adj Close'] - 1

# Print the final merged portfolio
print(merged_portfolio)

# HIGHEST VALUE loppuu

# PLOTS WITH PLOTLY
# Plot 1: Ticker return over time vs OMXH 25 (in local currency and percentage)
trace1 = go.Bar(
    x=merged_portfolio['Ticker'][0:10],
    y=merged_portfolio['Stock Gain / Loss'][0:10],
    name='Ticker Total Return (Local currency)')

trace2 = go.Bar(
    x=merged_portfolio['Ticker'][0:10],
    y=merged_portfolio['OMXH25 Gain / Loss'][0:10],
    name='OMXH 25 Total Return (Local currency)')

trace3 = go.Scatter(
    x=merged_portfolio['Ticker'][0:10],
    y=merged_portfolio['ticker return'][0:10],
    name='Ticker Total Return (in %)',
    yaxis='y2')

trace4 = go.Scatter(
    x=merged_portfolio['Ticker'][0:10],
    y=merged_portfolio['OMXH25 Return'][0:10],
    name='OMXH 25 Total Return (in %)',
    yaxis='y2')


data = [trace1, trace2, trace3, trace4]

layout = go.Layout(title='Total Return vs OMXH 25'
                   , barmode='group'
                   , yaxis=dict(title='Gain / Loss (Local currency)')
                   , yaxis2=dict(title='Return (%)', overlaying='y', side='right', tickformat=".2%")
                   , xaxis=dict(title='Ticker')
                   , legend=dict(x=.75, y=1)
                   )

fig_1 = go.Figure(data=data, layout=layout)
plot(fig_1)

# Plot 2: Cum investments and returns over time
trace1 = go.Bar(
    x=merged_portfolio['Ticker'],
    y=merged_portfolio['Cum Investments'],
    # mode = 'lines+markers',
    name='Cum Investments')

trace2 = go.Bar(
    x=merged_portfolio['Ticker'],
    y=merged_portfolio['Cum Ticker Returns'],
    # mode = 'lines+markers',
    name='Cum Ticker Returns')

trace3 = go.Scatter(
    x=merged_portfolio['Ticker'],
    y=merged_portfolio['Cum Ticker ROI Mult'],
    # mode = 'lines+markers',
    name='Cum ROI  Mult'
    , yaxis='y2')

data = [trace1, trace2, trace3]

layout = go.Layout(title='Total Cumulative Investments Over Time'
                   , barmode='group'
                   , yaxis=dict(title='Returns')
                   , xaxis=dict(title='Ticker')
                   , legend=dict(x=.4, y=1)
                   , yaxis2=dict(title='Cum ROI Mult', overlaying='y', side='right')
                   )

fig = go.Figure(data=data, layout=layout)
plot(fig)

# Plot 3: OMXH 25 vs Ticker price development comparisons
chart_tickers = portfolio_df['Ticker'].unique()
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

# Make (?)

# Plot 3
trace1 = go.Scatter(
    x=chart_data_eval_pivot_02_filled['Date'],
    y=chart_data_eval_pivot_02_filled['AAPL'],
    mode='lines',
    name='AAPL Price'
)

data = [trace1]

layout = go.Layout(title='Share Price development'
                   , barmode='group'
                   , yaxis=dict(title='Price (indexed)')
                   , xaxis=dict(title='Ticker')
                   , legend=dict(x=1, y=1)
                   )

fig = go.Figure(data=data, layout=layout)

for x in chart_tickers[1:]:
    fig.add_trace(go.Scatter(x=chart_data_eval_pivot_02_filled['Date'],
                             y=chart_data_eval_pivot_02_filled[x],
                             mode='lines',
                             name=x + ' Price'
                             ))
plot(fig)

# Opacity to the start that changes to color on the time of acquisition

# Plot 4: Pie chart for sector values
# Calculate market cap per sector in a pivot
pie_pivot = pd.pivot_table(merged_portfolio,
                           index = 'Sector',
                           values = 'Ticker Share Value',
                           aggfunc = 'sum')
pie_pivot.reset_index(inplace=True)

trace = go.Pie(
    labels=pie_pivot['Sector'],
    values=pie_pivot['Ticker Share Value'],
)
plot([trace])

# Plot 5: Pie chart for geographical allocation
# Calculate market cap per sector in a pivot
pie_pivot_2 = pd.pivot_table(merged_portfolio,
                           index = 'Country',
                           values = 'Ticker Share Value',
                           aggfunc = 'sum')
pie_pivot_2.reset_index(inplace=True)

trace_2 = go.Pie(
    labels=pie_pivot_2['Country'],
    values=pie_pivot_2['Ticker Share Value'],
)
plot([trace_2])


# PLOTS WITH PLOTLY loppuu

# DATA OUTPUT
# Data outputs
# Write to file
os.getcwd()  # Check the home directory
merged_portfolio.to_csv('analyzed_portfolio.csv', index=None)

chart_tickers_df = pd.DataFrame(chart_tickers)
chart_tickers_df.columns=['Ticker']
chart_tickers_df.to_csv('tickers.csv', header=True, index=None)



# merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers = merged_portfolio_omxh_latest_YTD_omxh_closing_high[
#     ['Ticker']]
# 
# merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers = merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers.drop_duplicates(
#     ['Ticker'], keep='first')
# 
# merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers = merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers[
#     'Ticker'].unique()
# 
# merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers = merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers.tolist()
# 
# merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers.append('OMXH25')
# 
# merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers = pd.DataFrame(
#     data=merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers, columns=['Ticker'])
# 
# merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers.sort_values(by='Ticker', ascending=True, inplace=True)
# 
# merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers.head()
# 
# # Write to file
# os.getcwd()  # Check the home directory
# merged_portfolio.to_csv('analyzed_portfolio.csv')
# merged_portfolio['Ticker'].unique().to_csv('tickers.csv')
# 
# # Create tickers to be used in the dashboard's dropdown
# merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers = merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers[
#     'Ticker'].unique()
# 
# merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers = merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers.tolist()
# 
# merged_portfolio_omxh_latest_YTD_omxh_closing_high_tickers

# DATA OUTPUT loppuu