#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:43:03 2025

@author: rrodriguesbend
"""

# %% Importing libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors
import matplotlib.gridspec as gridspec
from scipy import stats
import pyvinecopulib as pv
from scipy.stats import rankdata
from matplotlib.colors import Normalize
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import colormaps
from matplotlib import rc
from pyextremes import get_extremes
from pyextremes import EVA
from scipy.stats import genextreme, genpareto, weibull_min, expon, gamma, gumbel_r, beta
from IPython.display import display
import random

random.seed(1) 
# Empirical CDF
def ecdf(data):
    sorted_data = np.sort(data)
    cdf = np.zeros_like(sorted_data, dtype=float)
    for ii in range(len(sorted_data)):
        cdf[ii] = (ii + 1) / (len(sorted_data) + 1)
    return sorted_data, cdf
# %% Read data 
Data0 = pd.read_csv('Data/Data_Falsterbo10195_YSTAD.csv', 
                   parse_dates=['Time'])
Data0.set_index('Time', inplace=True)
MSL          = 14.3-(-0.07)*(2020-2000) # MSL in RH2000 (cm)
Data0['SSH'] = (Data0['SSH'] + MSL)/ 100 

# %%  Selecting per direction
ID_Dir  = (Data0.loc[Data0.index, 'Thetap'] >= 270) | (Data0.loc[Data0.index, 'Thetap'] < 45)
DataDir = Data0.loc[Data0.index[ID_Dir]].copy()
Threshold = np.quantile(DataDir['Hm0'], 0.995)

# %% POT 
Var        = DataDir['Hm0']
Threshold  = np.quantile(Var,0.995)
DeclusTime = 12

extremes = get_extremes(Var, "POT", threshold=Threshold, 
                                r=f"{DeclusTime}h")
Data = DataDir.loc[extremes.index]
# ---
# sns.pairplot(Data)

Time    = Data.index         # Time []
Wl      = Data['SSH']        # Sea Level [m + RH2000]
Hs      = Data['Hm0']        # Hm0 [m]
Tp      = Data['Tp']         # Tp [sec]
WaveDir = Data['Thetap']     # Dir [deg]
WindSpd = Data['WindSpeed']  # Wind Speed [m/s]
WindDir = Data['WindDir']    # Wind Dir [deg]

# %% Previsualization
# Set the color range and number of discrete colors
vmin, vmax = 0, 360  # Start and end values for the color scale
num_colors = 8  # Number of discrete colors
# Define discrete color levels
boundaries = np.linspace(vmin, vmax, num_colors + 1)
tick_interval = (vmax - vmin) / num_colors
ticks = boundaries  # Use boundaries directly as tick positions
# Create a discrete colormap
cmap = plt.cm.seismic  # Base colormap
norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
# Create the figure and axis
fig   = plt.figure( figsize = (8,5))
gs    = gridspec.GridSpec( ncols = 1, nrows = 1, figure = fig)
ax1   = fig.add_subplot( gs[0, 0] )
scatter = ax1.scatter(x=Wl, y=Hs, c=WaveDir, s=80, cmap=cmap, norm=norm, edgecolor='k')
cbar = plt.colorbar(scatter, ax=ax1, boundaries=boundaries, ticks=ticks)
cbar.set_label("Peak Wave Direction [deg]", fontsize=12)
ax1.set_xlabel(r'Water Level [m + RH2000]', fontsize=12)
ax1.set_ylabel(r'$H_{m0}$ [m]', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# --- Fitting Distributions for HS ---
POT_var  = Hs

# POT 
# Calculate Lambda
ny      = len(Data0.index)/(24*365)
Nt      = len(POT_var)
lambda_ = Nt / ny

# Fit Distributions
gev_params_hs = genextreme.fit(POT_var)  # GEV fit
gpd_params_hs = genpareto.fit(POT_var - Threshold, floc=0, scale=Threshold)  # GPD fit above threshold
gpd_params_hs = gpd_params_hs[:1] + (Threshold,) + gpd_params_hs[2:]

weibull_params = weibull_min.fit(POT_var)
expon_params = expon.fit(POT_var)
gamma_params = gamma.fit(POT_var)
gumbel_params = gumbel_r.fit(POT_var)
beta_params = beta.fit(POT_var)

x, f = ecdf(POT_var)
f[f == 1] = 0.999999
rp = 1 / ((1 - f) * lambda_)

# Define return period range
Rp = np.concatenate([np.arange(round(rp.min(), 2) + 0.01, 2, 0.01), np.arange(2, 20001)])
YPlot = 1 - 1 / (Rp * lambda_)

# Compute return values
RV_gpd = genpareto.ppf(YPlot, *gpd_params_hs)
RV_gev = genextreme.ppf(YPlot, *gev_params_hs)
RV_weibull = weibull_min.ppf(YPlot, *weibull_params)
RV_expon = expon.ppf(YPlot, *expon_params)
RV_gamma = gamma.ppf(YPlot, *gamma_params)
RV_gumbel = gumbel_r.ppf(YPlot, *gumbel_params)
RV_beta = beta.ppf(YPlot, *beta_params)

# Plot HS
plt.figure(figsize=(5, 4))
plt.plot(Rp, RV_gpd, label='Generalized Pareto', color='red', linewidth=2)
plt.plot(Rp, RV_gev, label='Generalized Extreme Value', color='black', linewidth=2)
plt.plot(Rp, RV_weibull, label='Weibull', color='blue', linewidth=2)
plt.plot(Rp, RV_expon, label='Exponential', color='green', linewidth=2)
plt.plot(Rp, RV_gamma, label='Gamma', color='purple', linewidth=2)
plt.plot(Rp, RV_gumbel, label='Gumbel', color='orange', linewidth=2)
plt.plot(Rp, RV_beta, label='Beta', color='brown', linewidth=2)
plt.scatter(rp, x, label='Empirical Data', color='gray', edgecolor='k', alpha=0.7, zorder=10,linewidth=0.5)
plt.xscale('log')
plt.xlim([min(rp)-1, 1000])
plt.ylim([Threshold-0.1, 4])
plt.xlabel('TR [years]', fontweight='bold')
plt.ylabel('Hm0 [m]', fontweight='bold')
plt.grid(True, which="both", linestyle='dotted', alpha=0.7)
plt.legend(loc='upper left', fontsize=7)
plt.title('Return Period for Hm0', fontsize=10)
plt.tight_layout()
plt.show()

# %%# %% Goodness of fitting using PDF
# Calculate AIC, BIC, RMSE, and R^2 for each distribution
results = []

distributions = {
    "Generalized Pareto": (genpareto, gpd_params_hs),
    "Generalized Extreme Value": (genextreme, gev_params_hs),
    "Weibull": (weibull_min, weibull_params),
    "Exponential": (expon, expon_params),
    "Gamma": (gamma, gamma_params),
    "Gumbel": (gumbel_r, gumbel_params),
    "Beta": (beta, beta_params)
}

for name, (dist, params) in distributions.items():
    # Compute the PPF (quantiles)
    fitted_ppf = dist.ppf(f, *params)  # PPF at empirical CDF values
    fitted_ppf = np.maximum(fitted_ppf, 1e-10)  # Avoid any numerical issues

    # Log-likelihood using PPF
    differences = np.abs(x - fitted_ppf)
    log_likelihood = -np.sum(np.log(differences + 1e-10))  # Small epsilon for stability

    # Number of parameters
    k = len(params)

    # AIC and BIC
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(len(x)) - 2 * log_likelihood

    # RMSE (between empirical and fitted quantiles)
    rmse = np.sqrt(np.mean(differences ** 2))

    # R^2 (coefficient of determination)
    ss_total = np.sum((x - np.mean(x)) ** 2)  # Total sum of squares
    ss_residual = np.sum(differences ** 2)  # Residual sum of squares
    r_squared = 1 - (ss_residual / ss_total)  # Coefficient of determination

    # Append results
    results.append({
        "Distribution": name,
        "AIC": aic,
        "BIC": bic,
        "RMSE": rmse,
        "R^2": r_squared
    })

# Convert results to a DataFrame for easy visualization
results_df = pd.DataFrame(results)
results_df.sort_values(by="AIC", inplace=True)

# Display the results
print(results_df)
# %% --- Fitting Distributions for WL ---

POT_var  = Wl

# POT 
# Calculate Lambda
ny      = len(Data0.index)/(24*365)
Nt      = len(POT_var)
lambda_ = Nt / ny

# Fit Distributions
gev_params = genextreme.fit(POT_var)  # GEV fit

weibull_params = weibull_min.fit(POT_var)
expon_params = expon.fit(POT_var)
gamma_params = gamma.fit(POT_var)
gumbel_params = gumbel_r.fit(POT_var)
beta_params = beta.fit(POT_var)

x, f = ecdf(POT_var)
f[f == 1] = 0.999999
rp = 1 / ((1 - f) * lambda_)

# Define return period range
Rp = np.concatenate([np.arange(round(rp.min(), 2) + 0.01, 2, 0.01), np.arange(2, 20001)])
YPlot = 1 - 1 / (Rp * lambda_)

# Compute return values
RV_gev = genextreme.ppf(YPlot, *gev_params)
RV_weibull = weibull_min.ppf(YPlot, *weibull_params)
RV_expon = expon.ppf(YPlot, *expon_params)
RV_gamma = gamma.ppf(YPlot, *gamma_params)
RV_gumbel = gumbel_r.ppf(YPlot, *gumbel_params)
RV_beta = beta.ppf(YPlot, *beta_params)

# Plot HS
plt.figure(figsize=(5, 4))
plt.plot(Rp, RV_gev, label='Generalized Extreme Value', color='black', linewidth=2)
plt.plot(Rp, RV_weibull, label='Weibull', color='blue', linewidth=2)
plt.plot(Rp, RV_expon, label='Exponential', color='green', linewidth=2)
plt.plot(Rp, RV_gamma, label='Gamma', color='purple', linewidth=2)
plt.plot(Rp, RV_gumbel, label='Gumbel', color='orange', linewidth=2)
plt.plot(Rp, RV_beta, label='Beta', color='brown', linewidth=2)
plt.scatter(rp, x, label='Empirical Data', color='gray', edgecolor='k', alpha=0.7, zorder=10,linewidth=0.5)
plt.xscale('log')
plt.xlim([min(rp)-1, 1000])
plt.ylim([min(x)-0.1, 2.5])
plt.xlabel('TR [years]', fontweight='bold')
plt.ylabel('Water Level [m + RH2000]', fontweight='bold')
plt.grid(True, which="both", linestyle='dotted', alpha=0.7)
plt.legend(loc='upper left', fontsize=7)
plt.title('Return Period for Water Level', fontsize=10)
plt.tight_layout()
plt.show()

# %% Goodness of fitting 
# Calculate AIC, BIC, RMSE, and R^2 for each distribution
results = []

distributions = {
    "Generalized Extreme Value": (genextreme, gev_params_hs),
    "Weibull": (weibull_min, weibull_params),
    "Exponential": (expon, expon_params),
    "Gamma": (gamma, gamma_params),
    "Gumbel": (gumbel_r, gumbel_params),
    "Beta": (beta, beta_params)
}

for name, (dist, params) in distributions.items():
    # Compute the PPF (quantiles)
    fitted_ppf = dist.ppf(f, *params)  # PPF at empirical CDF values
    fitted_ppf = np.maximum(fitted_ppf, 1e-10)  # Avoid any numerical issues

    # Log-likelihood using PPF
    differences = np.abs(x - fitted_ppf)
    log_likelihood = -np.sum(np.log(differences + 1e-10))  # Small epsilon for stability

    # Number of parameters
    k = len(params)

    # AIC and BIC
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(len(x)) - 2 * log_likelihood

    # RMSE (between empirical and fitted quantiles)
    rmse = np.sqrt(np.mean(differences ** 2))

    # R^2 (coefficient of determination)
    ss_total = np.sum((x - np.mean(x)) ** 2)  # Total sum of squares
    ss_residual = np.sum(differences ** 2)  # Residual sum of squares
    r_squared = 1 - (ss_residual / ss_total)  # Coefficient of determination

    # Append results
    results.append({
        "Distribution": name,
        "AIC": aic,
        "BIC": bic,
        "RMSE": rmse,
        "R^2": r_squared
    })

# Convert results to a DataFrame for easy visualization
results_df = pd.DataFrame(results)
results_df.sort_values(by="AIC", inplace=True)

# Display the results
print(results_df)


# %% Fitting magins 
# --- Fitting Distributions for HS ---
Var1 = Hs
Var2 = Wl

# Calculate Lambda
ny      = len(Data0.index)/(24*365)
Nt1 = len(Var1)
Nt2 = len(Var2)
lambda1 = Nt1 / ny

# Fit Distributions
params_hs = expon.fit(Var1)
params_wl = gamma.fit(Var2) 

x1, f1 = ecdf(Var1)
x2, f2 = ecdf(Var2)

# Avoid division by zero
f1[f1 == 1] = 0.999999
f2[f2 == 1] = 0.999999
rp1 = 1 / ((1 - f1) * lambda1)
rp2 = 1 / ((1 - f2) * lambda1)

# Define return period range
Rp1 = np.concatenate([np.arange(round(rp1.min(), 2) + 0.01, 2, 0.01), np.arange(2, 20001)])
Rp2 = np.concatenate([np.arange(round(rp2.min(), 2) + 0.01, 2, 0.01), np.arange(2, 20001)])

# Compute return values
RV_hs = expon.ppf(1 - 1 / (Rp1 * lambda1), *params_hs)
RV_wl = gamma.ppf(1 - 1 / (Rp2 * lambda1), *params_wl)

# Plot Combined Results
fig   = plt.figure( figsize = (6,6))
gs    = gridspec.GridSpec( ncols = 2, nrows = 2, figure = fig)

ax11   = fig.add_subplot( gs[0, 0] ) 
ax12   = fig.add_subplot( gs[0, 1] )
ax22   = fig.add_subplot( gs[1, 1] )

# Top left: Return Period vs HS
ax11.plot(Rp1, RV_hs, '-k', linewidth=1.5, label='Exponential')
ax11.scatter(rp1, x1, s=25, color='gray', alpha=0.7, label='Empirical Data')
ax11.set_xscale('log')
ax11.set_xlim([0.1, 1000])
ax11.set_ylim([1.75, 3.5])
ax11.set_xlabel('RP [year]')
ax11.set_ylabel('Hm0 [m]')
ax11.grid(True, which='both', linestyle='dotted')
ax11.legend(loc = 'upper left')

# Top right: Scatter WL vs HS
ax12.scatter(Var2, Var1, s=25, color='gray', alpha=0.7)
ax12.set_xlim([-1, 1.0])
ax12.set_ylim([1.75, 3.5])
ax12.grid(True, which='both', linestyle='dotted')
ax12.set_xlabel('Water Level [m + RH2000]')
ax12.set_ylabel('Hm0 [m]')

# Bottom left: Return Period vs WL
ax22.plot(RV_wl, Rp2, '-k', linewidth=1.5, label='Gamma')
ax22.scatter(x2, rp2, s=25, color='gray', alpha=0.7, label='Empirical Data')
ax22.set_yscale('log')
ax22.set_ylim([0.1, 1000])
ax22.set_xlim([-1, 1.0])
ax22.set_xlabel('Water Level [m + RH2000]')
ax22.set_ylabel('RP [year]')
ax22.grid(True, which='both', linestyle='dotted')
ax22.legend(loc = 'upper left')

plt.tight_layout()
plt.show()





