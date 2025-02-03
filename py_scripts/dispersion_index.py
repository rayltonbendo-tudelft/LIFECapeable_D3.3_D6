# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:04:20 2025

@author: rrodriguesbend
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.signal import find_peaks

# Dispersion Index function with confidence intervals
def dispersion_index(observations, min_threshold, max_threshold, step, dl, alpha=0.05):
    """
    Calculate and plot the Dispersion Index (DI) for a range of thresholds.

    Parameters:
        observations (pd.Series): Time-series data to analyze.
        min_threshold (float): Minimum threshold value.
        max_threshold (float): Maximum threshold value.
        step (int): Number of threshold steps between min and max.
        dl (int): Minimum distance between peaks (in hours).
        alpha (float): Significance level for confidence intervals.

    Returns:
        None
    """
    thresholds = np.linspace(min_threshold, max_threshold, step)
    e_mean = []
    var_mean = []
    M = len(observations.index.year.unique())  # Number of years in the sample

    for th in thresholds:
        # Find peaks exceeding the threshold
        peaks, _ = find_peaks(observations, height=th, distance=dl)
        excesses = observations.iloc[peaks] - th
        years = excesses.index.year.unique()
        n_excesses = []

        for year in years:
            # Count excesses for each year
            yearly_excesses = excesses[excesses.index.year == year]
            n_excesses.append(len(yearly_excesses))

        # Calculate mean and variance of excess counts
        e_mean.append(np.mean(n_excesses))
        var_mean.append(np.var(n_excesses))

    di_values = np.array(e_mean) / np.array(var_mean)

    # Confidence intervals for the Dispersion Index
    ci_lower = chi2.ppf(alpha / 2, M - 1) / (M - 1)
    ci_upper = chi2.ppf(1 - alpha / 2, M - 1) / (M - 1)

    # Plot Dispersion Index with confidence intervals
    plt.plot(thresholds, di_values, color='red',linestyle='-', marker='none', label="Dispersion Index",linewidth = 2, alpha=0.75)
    plt.axhline(ci_lower, color='blue', linestyle='--', label=f"CI Lower ({ci_lower:.2f})", alpha=0.5)
    plt.axhline(ci_upper, color='blue', linestyle='--', label=f"CI Upper ({ci_upper:.2f})", alpha=0.5)
    plt.axhline(1, color='blue', linestyle='-', label= "Unit value ")
    plt.fill_between(thresholds, ci_lower, ci_upper, color='blue', alpha=0.1, label="Confidence Interval")
    plt.title("Dispersion Index Plot", fontsize=14, fontname='Calibri')
    plt.xlabel("Threshold", fontsize=12, fontname='Calibri')
    plt.ylabel("DI", fontsize=12, fontname='Calibri')
    plt.legend(fontsize=12)
    plt.grid(True)
