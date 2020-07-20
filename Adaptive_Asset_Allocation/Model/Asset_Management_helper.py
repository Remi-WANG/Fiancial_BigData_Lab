import pymysql
from pandas_datareader import data as web
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 5)
import scipy.optimize as sco
import warnings
# warnings.filterwarnings('ignore')


def dataloader(tickers, api, start, end, panel = False):
    '''
    tickers: a list of str, the ticker symbol(s) of the assets to download
    api: a str, name of one api that is accessible through pandas_datareader
    start: a str with format 'year-month-date', start date of the series
    end: a str with format 'year-month-date', end date of the series

    load financial time series data from some api and check NA
    '''
    if panel:
        data = web.DataReader(tickers, api, start, end)

    else:
        panel_data = web.DataReader(tickers, api, start, end)
        price = input('Please select a price type from the followings {}:'.format(panel_data.columns))
        data = panel_data[price]

    if data.isnull().values.any():
        warnings.warn('Your data contains some NA, which may be an issue. We suggest to check and deal with it.')
    else:
        print('Successfully loaded data')

    return data

def dataslicer(data, length, reverse = False, start = None, end = None):
    '''
    data: a dataframe whose index is str or datetime with format '%Y-%m-%d'
    length: a int, length of the sliced dataframe; if None then the slicer is to filter a piece of data enclosed by start and end
    reverse: a Boolean value, if true then the period is data[:length]; else data[-length:]
    start: a str with format '%Y-%m-%d', indicating the start date of the sliced dataframe
    end: a str with format '%Y-%m-%d',indicating the end date of the sliced dataframe

    
    slice the dataframe either by a defined length or by a defined period
    '''
    if length is not None:
        if reverse:
            sliced_data = data[:length]
        else:
            sliced_data = data[-length:]
    else:
        sliced_data = data.loc[start:end]

    return sliced_data



class EqualWeight:
    def __init__(self, prices, n_reb):
        self.prices = prices
        self.n_reb = n_reb
        self.length = len(prices)
        self.width = len(prices.columns)
        self.returns = np.log(prices / prices.shift(1)).fillna(0)
        
    def cal_weights(self, output = False):
        weights = pd.DataFrame([[1. / self.prices.iloc[date].count()] * len(self.prices.columns) for date in range(self.length)],\
                       index=self.prices.index, columns=self.prices.columns)
        weights = rebalance(weights, self.n_reb) 

        if output:
            rebalance_times = self.length // self.n_reb
            short_weights = weights.iloc[[i for i in range(0, (rebalance_times +1) * self.n_reb,self.n_reb)]]
            short_weights.to_csv('EqualWeight_weights.csv')

        return weights

class RiskParity:
    def __init__(self, prices, n_reb, n_vol):
        self.prices = prices
        self.n_reb = n_reb
        self.n_vol = n_vol
        self.length = len(prices)
        self.width = len(prices.columns)
        self.returns = np.log(prices / prices.shift(1)).fillna(0)

    def cal_hist_vol(self):
        return self.returns.rolling(self.n_vol).std().fillna(1)

    def cal_weights(self, output = False):
        '''
        output: if ot not to output the weights dataframe where only the weights on rebalance days are kept

        '''
        inverse_vol = 1/ self.cal_hist_vol()
        weights = pd.DataFrame([np.multiply(inverse_vol.iloc[:,index], 1 / inverse_vol.sum(axis=1)) for index in range(self.width)], \
            index=self.prices.columns, columns=self.prices.index).T
        weights = rebalance(weights, self.n_reb) 

        if output:
            short_weights = weights.iloc[[i for i in range(0, self.length ,self.n_reb)]]
            short_weights.to_csv('RiskParity_weights.csv')

        return weights

class EqualMomentum:
    def __init__(self, prices, n_reb, n_vol, n_mom, n_top):
        self.prices = prices
        self.n_reb = n_reb
        self.n_mom = n_mom
        self.n_top = n_top
        self.length = len(prices)
        self.width = len(prices.columns)
        self.returns = np.log(prices / prices.shift(1)).fillna(0)

    def cal_momentum(self):
        np.random.seed(1)
        momentum = self.prices/ self.prices.shift(self.n_mom) 
        # Use random values to fill na
        len_na = len(momentum[momentum.isnull().any(axis=1)])
        momentum[:len_na] = np.random.randn((len_na), len(momentum.columns))

        return momentum

    def cal_rankings(self):
        '''
        get the momentum descending ranking dataframe 
        '''
        momentum = self.cal_momentum()
        return momentum.rank(axis=1, ascending=False)

    def cal_weights(self, output = False):
        rankings = self.cal_rankings()
        weights = ntop(rankings, self.n_top)
        weights = rebalance(weights, self.n_reb) 
        
        if output:
            short_weights = weights.iloc[[i for i in range(0, self.length ,self.n_reb)]]
            short_weights.to_csv('EqualMomentum_weights.csv')

        return weights

class RiskParMomentum:
    def __init__(self, prices, n_reb, n_vol, n_mom, n_top):
        self.prices = prices
        self.n_reb = n_reb
        self.n_vol = n_vol
        self.n_mom = n_mom
        self.n_top = n_top
        self.length = len(prices)
        self.width = len(prices.columns)
        self.returns = np.log(prices / prices.shift(1)).fillna(0)

    def cal_momentum(self):
        np.random.seed(1)
        momentum = self.prices/ self.prices.shift(self.n_mom) 
        # Use random values to fill na
        len_na = len(momentum[momentum.isnull().any(axis=1)])
        momentum[:len_na] = np.random.randn((len_na), len(momentum.columns))

        return momentum

    def cal_rankings(self):
        '''
        get the momentum descending ranking dataframe 
        '''
        momentum = self.cal_momentum()
        return momentum.rank(axis=1, ascending=False)

    def cal_hist_vol(self):
        return self.returns.rolling(self.n_vol).std().fillna(1)

    def cal_weights(self, output = False):
        inverse_vol = 1/ self.cal_hist_vol()
        rankings = self.cal_rankings()
        w = inverse_vol * ntop(rankings, self.n_top)
        weights = pd.DataFrame([np.multiply(w.iloc[:,index], 1 / w.sum(axis=1)) for index in range((self.width))], \
        index=self.prices.columns, columns=self.prices.index).T
        weights = rebalance(weights, self.n_reb) 

        if output:
            short_weights = weights.iloc[[i for i in range(0, self.length ,self.n_reb)]]
            short_weights.to_csv('RiskParMomentum_weights.csv')

        return weights

class MinVarMomentum:
    def __init__(self, prices, n_reb, n_vol, n_mom, n_top):
        self.prices = prices
        self.n_reb = n_reb
        self.n_vol = n_vol
        self.n_mom = n_mom
        self.n_top = n_top
        self.length = len(prices)
        self.width = len(prices.columns)
        self.returns = np.log(prices / prices.shift(1)).fillna(0)

    def cal_momentum(self):
        np.random.seed(1)
        momentum = self.prices/ self.prices.shift(self.n_mom) 
        # Use random values to fill na
        len_na = len(momentum[momentum.isnull().any(axis=1)])
        momentum[:len_na] = np.random.randn((len_na), len(momentum.columns))

        return momentum

    def cal_rankings(self):
        '''
        get the momentum descending ranking dataframe 
        '''
        momentum = self.cal_momentum()
        return momentum.rank(axis=1, ascending=False)


    def p_std(self, weights, cov_ret):
        '''
        calculate the standard deviation of portfolio log return 
        it is the objective function of minimum variance optimization problem
        '''
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_ret, weights))) * np.sqrt(252)
        
        return portfolio_std

    def min_variance(self, width, cov_ret, drop):
        '''
        solve the minimum variance optimization problem and get the optimal weight vector
        '''
        num_assets = width
        args = (cov_ret)
        # constraints: sum of weights - 1 = 0 / non top assets have no weights
        constraints = ([{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},{'type': 'eq', 'fun': lambda x: x[drop]}])   
        bound = (0.0,1.0)
        bounds = tuple(bound for asset in range(num_assets))
        result = sco.minimize(self.p_std, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def get_zero_column(self, dataframe, row):
        '''
        get the index of assets that will not be selected (weight = 0)

        '''

        zero_column = []
        for column in range(len(dataframe.columns)):
            if dataframe.iloc[row][column] == 0:
                zero_column.append(column)
            
        return zero_column

    def get_minVar_momentum_weights(self, output = False):
        '''
        Build up the optimal weight dataframe
        '''
        
        # global variables
        length = self.length
        width = self.width
        weights = pd.DataFrame(np.zeros((length, width)), index = self.prices.index, columns = self.prices.columns)
        cov_ret = self.prices.rolling(self.n_vol).cov()
    
        # get equally momentum weights: weights_mom

        rankings = self.cal_rankings()
        weights_mom = ntop(rankings, self.n_top)
        
        for date in range(length):
            # get indexes of assets that should not be selected on this date
            drop_assets = self. get_zero_column(weights_mom, date)
            # calculate optimal weights given the rolling covariance
            cov_ret_date = cov_ret[date*width: (date+1)*width]
            weights_date = [round(x,2) for x in self.min_variance(width,cov_ret_date,drop_assets)['x']]
        
            # append the weights for this date to weight dataframe
            weights.iloc[date] = weights_date

        weights = weights.fillna(1/width)
        weights = rebalance(weights, self.n_reb) 
        
        if output:
            short_weights = weights.iloc[[i for i in range(0, self.length ,self.n_reb)]]
            short_weights.to_csv('MinVarMomentum_weights.csv')

        return weights


def save_metrics(portfolios, strategy, capital, return_frame, weights) :
    '''
    portfolio: a dictionary, the name of portfolio
    strategy: a str, the name of strategy
    capital: a scalar, initial investment
    return_frame: a dataframe, daily log return of each asset
    weights: a dataframe, portofolio weights of this strategy
    
    build the dict 'portfolio' as a dict of dicts: 
    {{strategy 1}, {strategy 2},... }
    where dict strategy n is used to save performance metrics, like Sharp ratio and max_drawdown
    '''
    portfolios[strategy] = {}
    p_return_frame = cal_portfolio_return(weights, return_frame, strategy, output=False)
    p_value =  cal_portfolio_value(p_return_frame, capital)
    p_sharp_ls = cal_sharp_ratio(p_return_frame)
    portfolios[strategy]['Return'] = p_sharp_ls[0]
    portfolios[strategy]['Volatility'] = p_sharp_ls[1]
    portfolios[strategy]['SharpRatio'] = p_sharp_ls[2]
    portfolios[strategy]['Value'] = p_value.iloc[-1,0]
    portfolios[strategy]['PositiveRollingYears'] = cal_positive_rolling_years(p_return_frame)
    portfolios[strategy]['MaxMarkdown'] = cal_portfolio_drawdown(p_value).drawdown.min()
    
    return portfolios

def rebalance(weights, n_reb):
    '''
    weights: a dataframe showing daily portofolio weights
    n_reb: an int, rebalance frequency (number of days)
    
    rebalance the portfolio weights with a given frequency
    
    '''
    for i in range (0, len(weights), n_reb):
        if len(weights) - i >= n_reb:
            weights.iloc[i: i+ n_reb] = [weights.iloc[i]] * n_reb
        else:
            weights.iloc[i:] = [weights.iloc[i]] * (len(weights) - i)

    return weights

def cal_portfolio_return(weights, return_frame, strategy, log = False, output = False, ):
    '''
    weights: a dataframe saving daily portofolio weights
    return_frame: a dataframe of daily returns of all the assets, should be log return if log = True
    strategy: a str, name of current strategy
    output: a boolean, whether to output the result as a csv file
    
    generate the daily log return dataframe of a portofolio given the weights
    according to modern portfolio theory(MPT) by Markowitz, the portofolio return
    is calculated as the weighted sum of the individual assets' simple returns.
    '''

    if log:
        # p_12s means from portofolio's Log return to Simple return 
    	p_l2s = np.exp(return_frame)-1
    	p_return_frame = pd.DataFrame([np.dot(weights.iloc[date], p_l2s.iloc[date]) for date in range(len(p_l2s))],\
                       index=return_frame.index, columns=['portfolio_return'])
    else :
    	p_return_frame = pd.DataFrame([np.dot(weights.iloc[date], return_frame.iloc[date]) for date in range(len(return_frame))],\
                       index=return_frame.index, columns=['portfolio_return'])
    
    p_return_frame = np.log1p(p_return_frame) 
    if output:
        p_return_frame.to_csv(strategy + '_return'+ '.csv')
    return p_return_frame

def cal_portfolio_value(portfolio_return , capital):
    '''
    portfolio_return: a dataframe of a portfolio's log return time series
    capital: a number, the investor's initial capital
    '''
    portfolio_value = np.exp(portfolio_return.cumsum()) * capital
    portfolio_value.columns = ['portfolio_value']
    
    return portfolio_value

def cal_portfolio_drawdown(portfolio_value):
    '''
    portfolio_value: a dataframe of portfolio value time series
    
    A drawdown (DD) is the observed loss from a peak to a trough of a portfolio
    '''
    previous_peak = portfolio_value.cummax()
    drawdown = (portfolio_value - previous_peak) / previous_peak
    drawdown.columns = ['drawdown']
    
    print('maximum drawdown is {:.2%}'.format(drawdown.drawdown.min()))
    return drawdown

def cal_positive_rolling_years(portfolio_return):
    '''
    portfolio_return: a dataframe of a portfolio's log return time series
    
    get the proportion of positive rolling years (252 trading days per year)
    '''
    if len(portfolio_return) < 252:
        positive_rolling_year = 'NA'
    else:
        p_rolling_return = pd.DataFrame()
        p_rolling_return['rolling_return'] = portfolio_return['portfolio_return'].rolling(252).sum().dropna()
        positive_rolling_year = len(p_rolling_return.loc[p_rolling_return['rolling_return'] >= 0]) / len(p_rolling_return)
    
    print('positive rolling years is {}'.format(positive_rolling_year))
    return positive_rolling_year

def cal_sharp_ratio(portfolio_return):
    '''
    portfolio_return: a dataframe of a portfolio's daily log return time series
    
    calculate a portofolio's yearly sharp ratio through yearly return and yearly volatility
    '''
    pret = 252 * portfolio_return.portfolio_return.mean() 
    print('average yearly return is {:.2%}'.format(pret))
    pvol = np.sqrt(252) * portfolio_return.portfolio_return.std()
    print('yearly volatility is {:.2%}'.format(pvol))
    psharp = pret / pvol  
    print('yearly sharp ratio is {:.2%}'.format(psharp))
    return [i for i in [pret, pvol, psharp]]
    
def ntop(rankings, n):
    '''
    rankings: a dataframe of asset rolling performance rankings
    n: number of best-performaning assets to be selected for this date
    
    To generate the portfolio weights of simple momentum strategy
    top n assets share the same weights while the left have zero weights 
    '''
    weights = pd.DataFrame(0., index=rankings.index, columns=rankings.columns)
    for i in range(len(rankings)) :
        for s in rankings.columns :
            if rankings.iloc[i][s] <= n :
                weights.iloc[i][s] = 1. / n
            else :
                weights.iloc[i][s] = 0.
    
    return weights
