import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import permutations
from statsmodels.tsa.stattools import coint


class Fetch_Data :
    def __init__(self,start_date,end_date,tickers): 
        
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers 
        self.data = None
    def download_data(self): 
        self.data = yf.download(self.tickers, start = self.start_date, end = self.end_date)
        self.data = np.log(self.data)
        return self.data

class Select_Pair: 
    def __init__(self,data): 
        self.data = data
    
    def permutations(self) : 
        tickers = [i for i in self.data.columns]
        self.permutations = list(permutations(tickers,2))
        return self.permutations
    
    def are_cointegrated(self): 
        self.coint_results = {}
        for pair in self.permutations:
            ticker1 = pair[0]
            ticker2 = pair[1]

            price1 = self.data[ticker1]['Close']
            price2 = self.data[ticker2]['Close']

            score, pval, _ = coint(price1, price2)

            self.coint_results[(ticker1, ticker2)] = {'score': score, 'pvalue': pval}
        
        return None
        


        
        
    



