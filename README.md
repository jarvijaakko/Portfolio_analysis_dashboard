# A dashboard for analyzing investment portfolio performance

This repository contains two Python scripts that can be used to analyze an investment portfolio. 

As a starting point, the underlying portfolio needs to be submitted to the first script in the below format using e.g. MS Excel. This allows for easy modification: in case a new stock is bought, only the underlying Excel file needs to be updated and the scripts will take care of the rest. A mock portfolio is provided in the 'Stocks_v3.xls' file.

![Excel_01](https://user-images.githubusercontent.com/69734538/112727092-6ff9b900-8f29-11eb-924c-7fe2c9ac1806.png)

The first script 'Portfolio_analysis_03.py' will take the Excel file as input and perform a series of calculations to the portfolio. These include identifying all stocks and pulling data from Yahoo! Finance including a long historical period to analyze the stock performance. Ultimately, a final dataset is formed using numerous pandas DataFrame operations i.e. merging and pivoting data and written to a file 'analyzed_portfolio.csv'.

This file is taken as an input to the final script 'Portfolio_analysis_dash_03.py' that will produce the final HTML dashboard. This is done by plotting different metrics of interest from the portfolio using Plotly graphs and its Dash framework. Dash produces nice-looking, dynamic dashboards that allow easy zooming and variable exclusion, to name a few. In addition, all information is automatically updated according to the most recent stock market data to allow for up-to-date interpretation of portfolio performance.

The final dashboard looks like the following. The top left graph shows the returns for each stock given the period they've belonged to the portfolio vs the OMCH 25 index. The top right graph shows the cumulative return on equity (ROI) and the returns for each stock vs the investments made. On the bottom, I've indexed the price development of the stocks during the last year that would indicate the price trends within the portfolio. On the bottom right I've also included a few pie charts denoting the sectoral and geographical diversification of the given portfolio.

![dashboard_01](https://user-images.githubusercontent.com/69734538/112726996-ea760900-8f28-11eb-9d0b-7f71d238f269.png)

My future goals include incorporating this in some web platform, for example my homepage, to make the most out of this awesome application. I am also planning to merge these two scripts into one as it would allow for a more flexible usage.
