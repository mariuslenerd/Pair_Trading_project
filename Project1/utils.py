import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def plot_n_series(data,title,yscale,xlabel,ylabel) : 
    """
    function to plot n series simultaneously. I use a function such that all plots in the notebook will have the same format/style
    """
    plt.figure(figsize=(12,6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.yscale(yscale)
    plt.plot(data)

    return None


