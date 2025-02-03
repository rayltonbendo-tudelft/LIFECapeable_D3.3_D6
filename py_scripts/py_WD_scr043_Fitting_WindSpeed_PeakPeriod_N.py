# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:48:20 2025

@author: rrodriguesbend
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.optimize import curve_fit
from ci_delta_method import ci_delta_method
from predint import predint
from windrose import WindroseAxes

# %% Reading data 
Data = pd.read_csv('Data/Falsterbo_02_South_Wave10195_WD_POT_N_RH2000.csv',
                   parse_dates=['Time'])

Time    = Data['Time']       # Time []
Wl      = Data['SSH']        # Sea Level [m]
Hs      = Data['Hm0']        # Hm0 [m]
Tp      = Data['Tp']         # Tp [sec]
WaveDir = Data['Thetap']     # Dir [deg]
WindSpd = Data['WindSpeed']  # Wind Speed [m/s]
WindDir = Data['WindDir']    # Wind Dir [deg]
TWL     = Data['TWL']        # Total Water Level [m]


# %% Pre-visualisation 

# Set the color range and number of discrete colors
vmin, vmax = 0, 360  # Start and end values for the color scale
num_colors = 8  # Number of discrete colors

# Define discrete color levels
boundaries = np.linspace(vmin, vmax, num_colors + 1)
tick_interval = (vmax - vmin) / num_colors
ticks = boundaries  # Use boundaries directly as tick positions

# Create a discrete colormap
cmap = plt.cm.seismic  # Base colormap
Norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

# Create the figure and axis
fig   = plt.figure( figsize = (8,5))
gs    = gridspec.GridSpec( ncols = 1, nrows = 1, figure = fig)
ax1   = fig.add_subplot( gs[0, 0] )

# Create the scatter plot
scatter = ax1.scatter(x=Wl, y=Hs, c=WaveDir, s=80, cmap=cmap, norm=Norm, edgecolor='k')
# Add a colorbar with the discrete levels
cbar = plt.colorbar(scatter, ax=ax1, boundaries=boundaries, ticks=ticks)
cbar.set_label("Peak Wave Direction [deg]", fontsize=12)

# Add labels
ax1.set_xlabel(r'Sea level [m]', fontsize=12)
ax1.set_ylabel(r'$H_{m0}$ [m]', fontsize=12)

# Adjust layout
plt.tight_layout()
#list(colormaps)
# Show plot
plt.show()

# %% Defining Functions 
# Define the model function
# def model(x, a, b, c):
#     return a * x**b + c

def model(x, a, b):
    return a * x**b 
# %%
# ---------------------------- Modelling Tp ----------------------------- #

# Fit the curve
Tp_params, covariance = curve_fit(model, Hs, Tp, p0=[1, 1])  
# Extract the optimal parameters
a_opt, b_opt = Tp_params

# Residual standard deviation
residuals = Tp - model(Hs, *Tp_params)
residual_std = np.std(residuals)

Hs_fit = np.linspace(min(Hs), max(Hs), 100)
Tp_fit = model(Hs_fit, a_opt, b_opt)

#
ci_lower, ci_upper = predint(
    model, (Tp_params, covariance), Hs_fit,
    mode='observation', simultaneous='off', residual_std=residual_std
)

fig, ax = plt.subplots(figsize=(6, 5))
# sc = ax.scatter(Hs, Tp, s=50,label="Data", c=WaveDir, cmap=plt.cm.coolwarm, 
#                 alpha=0.7, edgecolors='k')
sc = ax.scatter(Hs, Tp, s=50,label="Data", color = 'gray',
                alpha=0.7, edgecolors='k')

plt.fill_between(Hs_fit, ci_lower, ci_upper, color='blue', alpha=0.12, 
                 label="95% CI", linestyle = '--')
# plt.fill_between(Hs_fit, ci_lower_coef, ci_upper_coef, color='green', alpha=0.3,
                 # label="95% Confidence Interval (Coefficients)")

ax.plot(Hs_fit, Tp_fit, label=f"Fit: y = {a_opt:.3f} * x^{b_opt:.3f}"
        , color="red", linewidth=2)

ax.set_ylabel("Tp [s]")
ax.set_xlabel("Hm0 [m]")
ax.set_ylim([np.nanmin(Tp) - 0.1, np.nanmax(Tp) + 0.1])  
# cbar = plt.colorbar(sc)
cbar.set_label("Wave Direction [°]", fontsize=12, labelpad=10)
ax.legend(loc="lower right")

#ax.grid()
plt.show()

# %% ------------------------- Modelling Wind Speed -------------------------- #
# Fit the curve
WS_params,Cov  = curve_fit(model, Hs, WindSpd, p0=[1, 1])  # Initial guesses
a_opt, b_opt = WS_params

# Residual standard deviation
residuals = WindSpd - model(Hs, *WS_params)
residual_std = np.std(residuals)

Hs_fit = np.linspace(min(Hs), max(Hs), 100)
WindSpd_fit = model(Hs_fit, *WS_params)

# CI
ci_lower, ci_upper = predint(
    model, (WS_params, Cov), Hs_fit,
    mode='observation', simultaneous='off', residual_std=residual_std
)


fig = plt.figure(figsize=(6, 5))
plt.scatter(Hs, WindSpd, label="Data", color = 'grey', alpha=0.7, edgecolors='k' )
plt.plot(Hs_fit, WindSpd_fit , label=f"Fit: y = {a_opt:.3f} * x^{b_opt:.3f}", color="red")
plt.fill_between(Hs_fit, ci_lower, ci_upper, color='blue', alpha=0.12, linestyle = '--',
                 label="95% CI")

plt.xlabel("Hm0 [m]")
plt.ylabel("Wind Speed [m/s]")
plt.legend(loc="lower right")
# cbar = plt.colorbar()
cbar.set_label("Wind Direction [°]", fontsize=12)
plt.show()

# %% 
# ---------------------------- Wave direction ----------------------------- #
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection="windrose")
ax.bar(WaveDir, Hs, bins=np.arange(0, 5, 1), normed=True, opening=0.8, 
        edgecolor='black',cmap=plt.cm.coolwarm)
# ax.set_legend(title="Hm0 [m]",loc="upper left")
ax.set_legend(title="Hm0 [m]", loc="center left", bbox_to_anchor=(1, 0.9))
#plt.title("Wind Rose from CSV Data")
plt.show()

# ---------------------------- Wind direction ----------------------------- #
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection="windrose")
ax.bar(WindDir, WindSpd, bins=np.arange(0, 21, 3), normed=True, opening=0.8, 
        edgecolor='black',cmap=plt.cm.viridis)
# ax.set_legend(title="Hm0 [m]",loc="upper left")
ax.set_legend(title="Wind Speed [m/s]", loc="center left", bbox_to_anchor=(1, 0.9))
#plt.title("Wind Rose from CSV Data")
plt.show()



