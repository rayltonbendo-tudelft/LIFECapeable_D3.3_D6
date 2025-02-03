#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:36:10 2025

@author: rrodriguesbend
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyextremes import get_extremes
from scipy.stats import kendalltau
from windrose import WindroseAxes
from extremalidx import extremalidx  # Import the function from the saved module
from pyextremes import plot_mean_residual_life, get_extremes
from runup import runup
from dispersion_index import dispersion_index

# Load data
Data = pd.read_csv('Data/Data_Falsterbo10195_YSTAD.csv', parse_dates=['Time'])
Data.set_index('Time', inplace=True)

# Total Water Level Calculation
MSL         = 14.3-(-0.07)*(2020-2000) # MSL in RH2000 (cm)
Data['SSH'] = (Data['SSH'] + MSL)/ 100 
Data['R2']  = runup(Data['Hm0'], Data['Tp'], beta_s=0.025) # Runup function based on Stockdon et al. (2006)
Data['TWL'] = Data['SSH'] + Data['R2']

Data = Data.dropna(subset=['TWL'])

# Threshold and Declustering Time Sensitivity
Thr = np.arange(0.99, 0.999, 0.001)
# DecTime = np.arange(1, 7)  # Declustering times (days)
DecTime = np.array((0.5, 1, 2, 3, 4, 5))  # Declustering times (days)

# Peak Over Threshold Analysis
extremes_storage = {jj: {kk: pd.DataFrame(columns=Data.columns) for kk in range(len(DecTime))} for jj in range(len(Thr))}
index_storage = {}

extremes_storage_E = {jj: {kk: pd.DataFrame(columns=Data.columns) for kk in range(len(DecTime))} for jj in range(len(Thr))}
index_storage_E = {}

extremes_storage_W = {jj: {kk: pd.DataFrame(columns=Data.columns) for kk in range(len(DecTime))} for jj in range(len(Thr))}
index_storage_W = {}

rho_Kendall = np.full((len(Thr), len(DecTime)), np.nan)
pval_Kendall = np.full((len(Thr), len(DecTime)), np.nan)
N_events = np.full((len(Thr), len(DecTime)), np.nan)
lambda_events = np.full((len(Thr), len(DecTime)), np.nan)

rho_Kendall_E = np.full((len(Thr), len(DecTime)), np.nan)
pval_Kendall_E = np.full((len(Thr), len(DecTime)), np.nan)
lambda_events_E = np.full((len(Thr), len(DecTime)), np.nan)

rho_Kendall_W = np.full((len(Thr), len(DecTime)), np.nan)
pval_Kendall_W = np.full((len(Thr), len(DecTime)), np.nan)
lambda_events_W = np.full((len(Thr), len(DecTime)), np.nan)

# Initialize Extremal Index storage
EI_omni = np.full((len(Thr), len(DecTime)), np.nan)
EI_E = np.full((len(Thr), len(DecTime)), np.nan)
EI_W = np.full((len(Thr), len(DecTime)), np.nan)

# Calculate observation period in years
# ny = (Data.index[-1] - Data.index[0]).days / 365.25
ny = len(Data.index)/(24*365)
ID_Dir  = (Data.loc[Data.index, 'Thetap'] >= 270) | (Data.loc[Data.index, 'Thetap'] < 45)
DataDir = Data.loc[Data.index[ID_Dir]].copy()

for jj, thr in enumerate(Thr):
    extremes_storage[jj] = {}
    index_storage[jj] = {}
    extremes_storage_E[jj] = {}
    index_storage_E[jj] = {}
    extremes_storage_W[jj] = {}
    index_storage_W[jj] = {}

    for kk, dec_time in enumerate(DecTime):
        extremes = get_extremes(DataDir['Hm0'], "POT", threshold=DataDir['Hm0'].quantile(thr), 
                                r=f"{dec_time}d")
        
        if not extremes.empty:
            # Omnidirectional
            extremes_storage[jj][kk] = Data.loc[extremes.index].copy()
            extremes_storage[jj][kk]['Extreme_Values'] = extremes
            index_storage[jj][kk] = extremes.index
            Hm0_values = DataDir.loc[extremes.index, 'Hm0'].to_numpy().flatten()
            ssh_values = DataDir.loc[(extremes.index), 'SSH'].to_numpy().flatten()
            
            time_values = DataDir.loc[extremes.index].index.to_numpy().astype('int64')
            N_events[jj, kk] = len(extremes)
            lambda_events[jj, kk] = N_events[jj, kk] / ny
            if len(Hm0_values) > 1 and len(ssh_values) > 1:
                rho, pval = kendalltau(Hm0_values, ssh_values)
                rho_Kendall[jj, kk] = rho
                pval_Kendall[jj, kk] = pval
                ei, _ = extremalidx(np.column_stack((time_values, ssh_values)))
                EI_omni[jj, kk] = ei

# %% Heat Maps Correlation 

# Plot settings
font_name = 'Calibri'
font_size = 12
caxis_rho = (-0.25, 0)  # Color axis limits for Kendall correlation
caxis_pval = (0, 0.05)  # Color axis limits for p-values
caxis_lambda = (0, 10)  # Color axis limits for number of events per year

# Heatmaps for Omnidirectional
fig, axes = plt.subplots(1, 3, figsize=(16, 7.0))

sns.heatmap(rho_Kendall, xticklabels=DecTime, yticklabels=Thr, cmap='Reds_r', 
            annot=True, ax=axes[0], vmin=caxis_rho[0], vmax=caxis_rho[1], 
            linewidths=0.5, linecolor='black')
axes[0].set_xlabel('Declustering Time [Days]', fontsize=font_size, fontname=font_name)
axes[0].set_ylabel('Threshold [quantile]', fontsize=font_size, fontname=font_name)
axes[0].set_title('Kendall Correlation', fontsize=font_size, fontname=font_name)

sns.heatmap(np.round(pval_Kendall, 3), xticklabels=DecTime, yticklabels=Thr, cmap='Blues', 
            annot=True, ax=axes[1], vmin=caxis_pval[0], vmax=caxis_pval[1], 
            linewidths=0.5, linecolor='black')
axes[1].set_xlabel('Declustering Time [Days]', fontsize=font_size, fontname=font_name)
axes[1].set_title('P-values', fontsize=font_size, fontname=font_name)

sns.heatmap(lambda_events, xticklabels=DecTime, yticklabels=Thr, cmap='Greys', 
            annot=True, ax=axes[2], vmin=caxis_lambda[0], vmax=caxis_lambda[1], 
            linewidths=0.5, linecolor='black')
axes[2].set_xlabel('Declustering Time [Days]', fontsize=font_size, fontname=font_name)
axes[2].set_title('Number of events per year', fontsize=font_size, fontname=font_name)

plt.suptitle('Omnidirectional', fontsize=font_size, fontname=font_name)
plt.tight_layout()
plt.show()

# %% Threshold & Declustering verification 
# Plotting Heatmaps for Extremal Index
# Omnidirectional
caxis_ei = (0, 0.5)  # Color axis limits for extremal index

fig, axes = plt.subplots(1, 1, figsize=(7, 7))
sns.heatmap(EI_omni, xticklabels=DecTime, yticklabels=np.round(Thr, 3), cmap='coolwarm', 
            annot=True, ax=axes, vmin=caxis_ei[0], vmax=caxis_ei[1]
            , fmt='.2f', cbar_kws={'label': 'Extremal Index'},
            linewidths=0.5, linecolor='black')
axes[0].set_xlabel('Declustering Time [Days]', fontsize=12, fontname='Calibri')
axes[0].set_ylabel('Threshold [quantile]', fontsize=12, fontname='Calibri')
axes[0].set_title('Extremal Index (Omnidirectional)', fontsize=12, fontname='Calibri')

# %% 
# Mean residual life
Var           = DataDir['Hm0']
MyThreshold   = np.quantile(Var, 0.995)
min_threshold = np.quantile(Var, 0.95)
max_threshold = np.quantile(Var, 0.999) * 1.16
DTime         = 12

# 
font_name = 'Calibri'
font_sizee = 14
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Plot Mean Residual Life
plot_mean_residual_life(Var, ax=ax1)
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
dispersion_index(observations=Var, min_threshold=min_threshold, 
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

# %% Time Series 
# Plot for Thetap analysis at Thr and DecTimE
IDThr = np.where(Thr == 0.995)[0][0]
IDDecTime = np.where(DecTime == 0.5)[0][0]
data_thr = extremes_storage[IDThr][IDDecTime]

font_size = 14
font_name = 'Calibri'
fig, axes = plt.subplots(4, 1,sharex=True, figsize=(15, 9))

axes[0].scatter(Data.index, Data['Hm0'],s= 10, color='gray',marker='o'
                , alpha = 0.7,linewidths=0.1,edgecolors='none', label='Data' )
axes[0].scatter(data_thr.index, data_thr['Hm0'],s= 30, color='b',marker='o'
                , alpha = 0.7,linewidths=0.1,edgecolors='none', label='Data' )
axes[0].axhline(y=DataDir['Hm0'].quantile(0.995), color='r', linestyle='--', 
                label='Threshold') 
axes[0].set_xlim(Data.index[0], Data.index[-1])
axes[0].set_ylabel('Hm0 [m]', fontsize=font_size, fontname=font_name)
axes[0].legend(fontsize=font_size*0.7,loc="upper right",ncol=3)

axes[1].scatter(Data.index, Data['Tp'],s= 10, color='gray',marker='o'
                , alpha = 0.7,linewidths=0.1,edgecolors='none', label='Data' )
axes[1].scatter(data_thr.index, data_thr['Tp'],s= 30, color='b',marker='o'
                , alpha = 0.7,linewidths=0.1,edgecolors='none', label='Data' )
axes[1].set_xlim(Data.index[0], Data.index[-1])
axes[1].set_ylabel('Tp [s]', fontsize=font_size, fontname=font_name)
# axes[1].legend(fontsize=font_size*0.7,loc="upper right",ncol=3)

axes[2].scatter(Data.index, Data['Thetap'],s= 10, color='gray',marker='o'
                , alpha = 0.7,linewidths=0.1,edgecolors='none', label='Data' )
axes[2].scatter(data_thr.index, data_thr['Thetap'],s= 30, color='b',marker='o'
                , alpha = 0.7,linewidths=0.1,edgecolors='none', label='Data', )
axes[2].set_xlim(Data.index[0], Data.index[-1])
axes[2].set_ylabel('Peak Wave Dir. [°]', fontsize=font_size, fontname=font_name)
# axes[2].legend(fontsize=font_size*0.7,loc="upper right",ncol=3)

axes[3].scatter(Data.index, Data['SSH'],s= 10, color='gray',marker='o'
                , alpha = 0.7,linewidths=0.1,edgecolors='none', label='Data' )
axes[3].scatter(data_thr.index, data_thr['SSH'],s= 30, color='b',marker='o'
                , alpha = 0.7,linewidths=0.1,edgecolors='none', label='Data', )
axes[3].set_xlim(Data.index[0], Data.index[-1])
axes[3].set_ylabel('Water Lelvel [m + RH2000]', fontsize=font_size, fontname=font_name)

plt.tight_layout()
plt.show()

# %% Scatter Plots
font_size = 14
font_name = 'Calibri'

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# ax1
axes[0, 0].scatter(Data['Hm0'], Data['Thetap'], s=10, color='gray', marker='o',
                   alpha=0.7,  label='Data')
axes[0, 0].scatter(DataDir['Hm0'], DataDir['Thetap'], s=30, color='k', marker='o',
                   alpha=0.7, label='East')
axes[0, 0].scatter(data_thr['Hm0'], data_thr['Thetap'], s=50, color='b', marker='o',
                   alpha=0.7, label='POT')
axes[0, 0].set_ylim(0, 360) 
axes[0, 0].set_yticks(np.arange(0, 361, 30)) 

axes[0, 0].set_xlabel('Hm0 [m]', fontsize=font_size, fontname=font_name)
axes[0, 0].set_ylabel('Peak Wave Dir [°]', fontsize=font_size, fontname=font_name)
axes[0, 0].legend(fontsize=font_size * 0.7, loc="upper right", ncol=3)

# ax2
axes[0, 1].scatter(Data['Hm0'], Data['Tp'], s=10, color='gray', marker='o',
                   alpha=0.7,  label='Data')
# axes[0, 1].scatter(DataDir['Hm0'], DataDir['Tp'], s=30, color='k', marker='o',
#                    alpha=0.7, label='East')
axes[0, 1].scatter(data_thr['Hm0'], data_thr['Tp'], s=50, color='b', marker='o',
                   alpha=0.7, label='POT')

axes[0, 1].set_xlabel('Hm0 [m]', fontsize=font_size, fontname=font_name)
axes[0, 1].set_ylabel('Tp [s]', fontsize=font_size, fontname=font_name)

# ax3
axes[1, 0].scatter(Data['Hm0'], Data['SSH'], s=10, color='gray', marker='o',
                   alpha=0.7,  label='Data')
# axes[1, 0].scatter(DataDir['Hm0'], DataDir['SSH'], s=30, color='k', marker='o',
#                    alpha=0.7, label='East')
axes[1, 0].scatter(data_thr['Hm0'], data_thr['SSH'], s=50, color='b', marker='o',
                   alpha=0.7, label='POT')

axes[1, 0].set_xlabel('Hm0 [m]', fontsize=font_size, fontname=font_name)
axes[1, 0].set_ylabel('Water Level [m + RH2000]', fontsize=font_size, fontname=font_name)

# ax4
axes[1, 1].scatter(Data['Hm0'], Data['WindSpeed'], s=10, color='gray', marker='o',
                   alpha=0.7,  label='Data')

axes[1, 1].scatter(data_thr['Hm0'], data_thr['WindSpeed'], s=50, color='b', marker='o',
                   alpha=0.7, label='POT')

axes[1, 1].set_xlabel('Hm0 [m]', fontsize=font_size, fontname=font_name)
axes[1, 1].set_ylabel('Wind Speed [m/s]', fontsize=font_size, fontname=font_name)

plt.tight_layout()
plt.show()

# %% Wave Rose Plots
fig = plt.figure(figsize=(6, 8))

# Omnidirectional Wave Rose
ax1 = fig.add_subplot(211, projection="windrose")
ax1.bar(data_thr['Thetap'], data_thr['Hm0'], bins=np.arange(0, 5, 1), normed=True, 
        opening=0.8, edgecolor='black',cmap=plt.cm.coolwarm)
ax1.set_legend(title="Hm0 [m]", loc="center left", bbox_to_anchor=(1, 0.9))
plt.title("Wave Rose")
plt.show()

plt.tight_layout()
plt.show()

# Wind Rose Plots
# Omnidirectional Wave Rose
ax1 = fig.add_subplot(212, projection="windrose")
ax1.bar(data_thr['WindDir'], data_thr['WindSpeed'], bins=np.arange(0, 21, 3), normed=True, 
        opening=0.8, edgecolor='black',cmap=plt.cm.viridis)
ax1.set_legend(title="Wind Speed [m]", loc="center left", bbox_to_anchor=(1, 0.9))
plt.title("Wind Rose")
plt.show()


plt.tight_layout()
plt.show()

# %% Save results
# data_thr.to_csv('Data/Falsterbo_02_South_Wave10195_WD_POT_N_RH2000.csv')





