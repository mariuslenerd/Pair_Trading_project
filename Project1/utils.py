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
    for col in data.columns : 
        plt.plot(data.index, data[col], label = col)
    plt.legend(title = 'Ticker', loc = 'best')
    plt.yscale('log')
    plt.grid()

    return None

def plot_wealth_positions_spread(data,residuals, threshold,positions,cumulative_pnl) : 
    ticker_A = data.columns[0]
    ticker_B = data.columns[1]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # Normalised spread + thresholds + shaded positions 
    axes[0].plot(residuals, color='gray', linewidth=1, label='z-score (spread)')
    axes[0].axhline( threshold, color='red',   linewidth=1.2, linestyle='--', label=f'Entry threshold (+{threshold})')
    axes[0].axhline(-threshold, color='green', linewidth=1.2, linestyle='--', label=f'Entry threshold (-{threshold})')
    axes[0].axhline(0,          color='black', linewidth=0.8)
    axes[0].set_title('Normalised Spread (z-score) & Positions', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('z-score')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(alpha=0.3)

    # Position sizes over time 
    axes[1].step(positions.index, positions[ticker_A], where='post', color='steelblue',
             linewidth=1.5, label=f'Position {ticker_A} (+1=long, -1=short)')
    axes[1].step(positions.index, positions[ticker_B], where='post', color='darkorange',
             linewidth=1.5, label=f'Position {ticker_B} (+β=long, -β=short)', linestyle='--')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Positions Over Time', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Position size')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(alpha=0.3)

    # Cumulative PnL 
    axes[2].plot(cumulative_pnl.index, cumulative_pnl, color='purple', linewidth=1.5, label='Cumulative net PnL')
    axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[2].fill_between(cumulative_pnl.index, cumulative_pnl, 0,
                     where=(cumulative_pnl >= 0), alpha=0.15, color='green', label='Profit')
    axes[2].fill_between(cumulative_pnl.index, cumulative_pnl, 0,
                     where=(cumulative_pnl < 0),  alpha=0.15, color='red',   label='Loss')
    axes[2].set_title('Cumulative PnL (fraction of notional)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Cumulative PnL')
    axes[2].set_xlabel('Date')
    axes[2].legend(loc='upper left', fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

 

