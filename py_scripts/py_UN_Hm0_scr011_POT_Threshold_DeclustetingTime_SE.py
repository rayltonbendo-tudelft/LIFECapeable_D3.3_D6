#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:26:33 2025

@author: rrodriguesbend
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import genextreme, genpareto, weibull_min, expon, gamma, gumbel_r, beta
from scipy.stats import rankdata, kendalltau
from pyextremes import get_extremes
from pyextremes import EVA
from pyextremes import plot_mean_residual_life
from pyextremes.plotting import plot_extremes
from pyextremes import plot_parameter_stability
from pyextremes import plot_return_value_stability
from pyextremes import plot_threshold_stability
from extremalidx import extremalidx  # Import the function from the saved module
from runup import runup
from dispersion_index import dispersion_index
import datetime

def ecdf(data):
    sorted_data = np.sort(data)
    cdf = np.zeros_like(sorted_data, dtype=float)
    for ii in range(len(sorted_data)):
        cdf[ii] = (ii + 1) / (len(sorted_data) + 1)
    return sorted_data, cdf

# %% Load data
Data0 = pd.read_csv('Data/Data_Falsterbo10958_YSTAD.csv', parse_dates=['Time'])
Data0.set_index('Time', inplace=True)

ID_Dir  = (Data0.loc[Data0.index, 'Thetap'] >= 45) & (Data0.loc[Data0.index, 'Thetap'] < 155)

Data = Data0[ID_Dir]


# %% Threshold and Declustering Time Sensitivity

Thr = np.arange(0.99, 0.999, 0.001)
# DecTime = np.arange(0.5, 5)  # Declustering times (days)
DecTime = np.linspace(0.5, 5, num = 10)

# Peak Over Threshold Analysis
extremes_storage = {jj: {kk: pd.DataFrame(columns=Data.columns) for kk in range(len(DecTime))} for jj in range(len(Thr))}
index_storage = {}

N_events = np.full((len(Thr), len(DecTime)), np.nan)
lambda_events = np.full((len(Thr), len(DecTime)), np.nan)

# Initialize Extremal Index storage
EI_omni = np.full((len(Thr), len(DecTime)), np.nan)

# Calculate observation period in years
ny = len(Data0.index)/(24*365)

for jj, thr in enumerate(Thr):
    extremes_storage[jj] = {}
    index_storage[jj] = {}

    for kk, dec_time in enumerate(DecTime):
        extremes = get_extremes(Data['Hm0'], "POT", threshold=Data['Hm0'].quantile(thr), 
                                r=f"{dec_time}d")
        
        if not extremes.empty:
            # Omnidirectional
            extremes_storage[jj][kk] = Data.loc[extremes.index].copy()
            extremes_storage[jj][kk]['Extreme_Values'] = extremes
            index_storage[jj][kk] = extremes.index
            Hm0_values = Data.loc[extremes.index, 'Hm0'].to_numpy().flatten()
            ssh_values = Data.loc[extremes.index, 'SSH'].to_numpy().flatten()
            time_values = Data.loc[extremes.index].index.to_numpy().astype('int64')
            # time_values = (time_values - time_values[0])/(24*3600)
            N_events[jj, kk] = len(extremes)
            lambda_events[jj, kk] = N_events[jj, kk] / ny
            if len(Hm0_values) > 1 and len(ssh_values) > 1:
                rho, pval = kendalltau(Hm0_values, ssh_values)
                ei, _ = extremalidx(np.column_stack((time_values, Hm0_values)))
                EI_omni[jj, kk] = ei

# %% Threshold & Declustering verification 
# Plotting Heatmaps for Extremal Index
# Omnidirectional
Yplot = np.quantile(Data['Hm0'], Thr) #np.round(Thr, 3)

caxis_ei = (0.5, 1)  # Color axis limits for extremal index
fig   = plt.figure( figsize = (13,9))
gs    = gridspec.GridSpec( ncols = 2, nrows = 1, figure = fig)
ax1   = fig.add_subplot( gs[0, 0] )

sns.heatmap(EI_omni, xticklabels=DecTime, yticklabels=np.round(Thr, 3), cmap='coolwarm', 
            annot=True,ax = ax1, vmin=caxis_ei[0], vmax=caxis_ei[1]
            , fmt='.2f', cbar_kws={'label': 'Extremal Index'},
            linewidths=0.5, linecolor='black')
ax1.set_xlabel('Declustering Time [Days]', fontsize=12, fontname='Calibri')
ax1.set_ylabel('Threshold [quantile]', fontsize=12, fontname='Calibri')
ax1.set_title('Extremal Index (Omnidirectional)', fontsize=12, fontname='Calibri')

ax2   = fig.add_subplot( gs[0, 1] )
sns.heatmap(lambda_events, xticklabels=DecTime, yticklabels=np.round(Yplot, 3), 
            cmap='Greys', annot=True,ax = ax2, vmin=0, vmax=10
            , fmt='.2f', cbar_kws={'label': 'Number of events per year'},
            linewidths=0.5, linecolor='black')
ax2.set_xlabel('Declustering Time [Days]', fontsize=12, fontname='Calibri')
ax2.set_ylabel('Threshold [m]', fontsize=12, fontname='Calibri')
ax2.set_title('Number of events per year (Omnidirectional)', fontsize=12, fontname='Calibri')
plt.tight_layout()
plt.show()

# %%
# Mean residual life
VAR           = Data['Hm0']
MyThreshold   = np.quantile(VAR, 0.995)
min_threshold = np.quantile(VAR, 0.90)
max_threshold = 3.5 #np.quantile(VAR, 0.999) * 1.16
DTime         = 12

# 
font_name = 'Calibri'
font_sizee = 14
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

# Plot Mean Residual Life
plot_mean_residual_life(VAR, ax=ax1)
ax1.axvline(x=round(MyThreshold, 2), color='black', linestyle='--', 
            label='Threshold', linewidth=2)
ax1.set_title("Mean Residual Life (MRL)", fontsize=font_sizee, fontname=font_name)
ax1.set_xlabel("Threshold", fontsize=font_sizee, fontname=font_name)
ax1.set_ylabel("Mean Excess", fontsize=font_sizee, fontname=font_name)
ax1.set_xlim(min_threshold, max_threshold)
ax1.legend(fontsize=font_sizee, loc='upper left')
ax1.grid(True, alpha=0.5)

# Plot Dispersion Index
plt.sca(ax2)  # Set current axis to ax2
dispersion_index(observations=VAR, min_threshold=min_threshold, 
                 max_threshold=max_threshold, step=100, dl=DTime)
ax2.axvline(x=round(MyThreshold, 2), color='black', linestyle='--', 
            label='Threshold', linewidth=2)

ax2.set_xlabel("Threshold", fontsize=font_sizee, fontname=font_name)
ax2.set_ylabel("Dispersion Index", fontsize=font_sizee, fontname=font_name)

ax2.set_xlim(min_threshold, max_threshold)
ax2.set_ylim(0.0, 4.5)
ax2.grid(True, alpha=0.5)

plt.title("Dispersion Index Plot", fontsize=font_sizee, fontname=font_name)
plt.tight_layout()
plt.show()

#%%

distributions = ["genpareto", "expon"]
stability_plot  = plot_threshold_stability(
    VAR,
    return_period=100,
    thresholds=np.linspace(1.5, 3.5, 20),
    distributions=distributions,
    r  = '12h',
)



