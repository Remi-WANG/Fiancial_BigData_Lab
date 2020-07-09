import pandas as pd
from pandas_datareader import data as web
import numpy as np
import matplotlib.pyplot as plt

ticker_ls = ['SPY','EFA','EWJ','EEM','IYR','RWX','IEF','TLT','DBC','GLD'] # list of asset tickers
start_date = "1995-01-01"
end_date = "2014-11-30"

# Talk with Yahoo Api
panel_data = web.DataReader(ticker_ls, 'yahoo', start_date, end_date)

