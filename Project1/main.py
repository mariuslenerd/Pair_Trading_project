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
        """
        Download data of interest using yahoo finance
        """
        self.data = yf.download(self.tickers, start = self.start_date, end = self.end_date)
        self.data = np.log(self.data)
        self.data = self.data['Close']
        return self.data

class Select_Pair: 
    def __init__(self,data): 
        self.data = data
    
    def permutations(self) : 
        """
        Extract all different possible permuations of pairs (ex : Booking - IHG, IHG-Booking,...)
        We need to test both ways because the fct cointegration first regresses Booking on IHG 
        (booking_t = alpha + beta IHG_t + epsilon_t) and then tests the residuals for stationarity. 

        The regression is the reason we cannot use one way to test cointegration, OLS is not symmetric (i.e regressing x on y is not the same as regressing y on x)
        this is due to the fact that beta_hat_y_knowing_x = Cov(x,y)/var(x) which is generally neq beta_hat_x_knowing_y = Cov(x,y)/Var(y)
        unless perfect linear correlation (i.e abs(rho) = 1). Also, a geometric explanation : ols minimizes vertical distance, not orthogonal distances
        """
        tickers = [i for i in self.data.columns]
        self.permutations = list(permutations(tickers,2))
        return self.permutations
    
    def are_cointegrated(self): 
        """
        Find the most cointegrated pair of assets. To do so, we employ the previously created pairs and 
        test each of them using the coint funciton from statsmodels.

        Note that we first restrict the length of the time series due to the fact that 1 asset has missing values until 2013. 
        To do so, we check what are the first and last value of each serie and force them to be the same (taking the maximum between 
        the 2 first values and the minimum between the 2 last values)
        """
        self.coint_results = {}
        for pair in self.permutations:
            ticker1 = pair[0]
            ticker2 = pair[1]

            price1 = self.data[ticker1].dropna()
            price2 = self.data[ticker2].dropna()

            start1 = price1.index[0]
            end1 = price1.index[-1]

            start2 = price2.index[0]
            end2 = price2.index[-1]

            if start1 != start2 : 
                    start = np.maximum(start1,start2)
            else : 
                    start = start1
            
            if end1 != end2 : 
                    end = np.minimum(end1,end2)
            else : 
                    end = end1

            price1 = price1[start:end]
            price2 = price2[start:end]

            score, pval, _ = coint(price1, price2)

            self.coint_results[(ticker1, ticker2)] = {'score': score, 'pvalue': pval}
        
        df_coint_results = pd.DataFrame(self.coint_results).transpose()

        self.most_coint_pair = df_coint_results['score'].idxmin()

        self.data_most_coint = self.data[[self.most_coint_pair[0], self.most_coint_pair[1]]].dropna()

        return self.most_coint_pair, self.data_most_coint


        
        
    



