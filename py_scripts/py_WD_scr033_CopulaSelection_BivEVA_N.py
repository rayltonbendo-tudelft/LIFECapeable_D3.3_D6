#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:46:51 2025

@author: rrodriguesbend
"""
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
from scipy.stats import genextreme, genpareto, weibull_min, expon, gamma, gumbel_r, beta
from scipy.optimize import curve_fit
from windrose import WindroseAxes
from scipy.stats import norm

import random

random.seed(1) 

# Empirical CDF
def ecdf(data):
    sorted_data = np.sort(data)
    cdf = np.zeros_like(sorted_data, dtype=float)
    for ii in range(len(sorted_data)):
        cdf[ii] = (ii + 1) / (len(sorted_data) + 1)
    return sorted_data, cdf

# %% Reading data 
Data = pd.read_csv('Data/Falsterbo_02_South_Wave10195_WD_POT_N_RH2000.csv',
                   parse_dates=['Time'])

Wl  = Data['SSH']
Hs  = Data['Hm0']

# %%
# =============================================================================
# Dependence. Let's start investigating the dependence between the obsevations. 
# To do so, you will:
# compute three measures of dependence: Pearson's , Spearman's, Kendall's tau
# transform the data into uniform marginals via the empirical cdf
# plot the scatterplot both in the original domain and in the unit square domain
# =============================================================================

# PEARSON rho
pearson = stats.pearsonr(Wl, Hs)
# Spearman rho
spearman = stats.spearmanr(Wl, Hs)
# Kendall tau
kendall = stats.kendalltau(Wl, Hs)

# Transform Marginals into uniform standard using empirical cdf
# NOTE: ecdf function returns sorted data. This will remove the dependence between u and v

u  = rankdata(Wl)/ (len(Wl) + 1)
v  = rankdata(Hs)/ (len(Hs) + 1)


fig   = plt.figure( figsize = (14,5))
gs    = gridspec.GridSpec( ncols = 2, nrows = 1, figure = fig)

ax1   = fig.add_subplot( gs[0, 0] )
ax2   = fig.add_subplot( gs[0, 1] )

scatter=ax1.scatter(Wl, Hs, s=100, alpha=1,marker="o", edgecolor='black')
ax1.set_xlabel('Water Level [m + RH2000]',fontsize=12)
ax1.set_ylabel('Hm0 [m]',fontsize=12)
ax1.set_title('Raw Data',fontsize=13)
ax1.grid(color = '.7', linestyle='dotted', zorder=-1)
ax1 = plt.gca()

scatter=ax2.scatter(u, v, s=100, alpha=1,marker="o", edgecolor='black')
ax2.set_xlabel('u',fontsize=12)
ax2.set_ylabel('v',fontsize=12)
ax2.set_title('Uniform Data',fontsize=13)
ax2.grid(color = '.7', linestyle='dotted', zorder=-1)
ax2 = plt.gca()

plt.show()
# %%
# =============================================================================
# List of all the copula families
# =============================================================================

cop_fam = []
# No need to rotate 
cop_fam.append(pv.BicopFamily.gaussian) 
cop_fam.append(pv.BicopFamily.student)
cop_fam.append(pv.BicopFamily.frank)
# You can rotate 
cop_fam.append(pv.BicopFamily.clayton)
cop_fam.append(pv.BicopFamily.gumbel)  
cop_fam.append(pv.BicopFamily.joe)
cop_fam.append(pv.BicopFamily.bb1)
cop_fam.append(pv.BicopFamily.bb6)
cop_fam.append(pv.BicopFamily.bb7)
cop_fam.append(pv.BicopFamily.bb8)

# =============================================================================
# AIC and BIC. Below, the function Bicop.fit is used to fit the copula families 
# listed above and calculate AIC and BIC. Based on the results, you can find a 
# plot of the best copula in the unit square domain and in the original domain 
# of the data. 
# =============================================================================

# Fit the above families and store AIC and BIC

FLOOD_u = np.empty([len(u), 2])
FLOOD_u[:, 0] = u
FLOOD_u[:, 1] = v

aic_bic_results = []

# Fit each copula family and calculate AIC and BIC
for f in range(3):
    cop_temp = pv.Bicop(family=cop_fam[f])
    cop_temp.fit(data=FLOOD_u)            
    aic = cop_temp.aic(u=FLOOD_u)          
    bic = cop_temp.bic(u=FLOOD_u)          
    aic_bic_results.append({"family": cop_fam[f].name, "rotation": 0,
                            "aic": aic, "bic": bic})

# For families with rotations
rotations = [0, 90, 180, 270]
for j in range(3, len(cop_fam)):
    for rotation in rotations:
        cop_temp = pv.Bicop(family=cop_fam[j], rotation=rotation)  
        cop_temp.fit(data=FLOOD_u)                               
        aic = cop_temp.aic(u=FLOOD_u)                            
        bic = cop_temp.bic(u=FLOOD_u)                             
        aic_bic_results.append({"family": cop_fam[j].name, "rotation": rotation,
                                "aic": aic, "bic": bic})

# Convert results to a table 
aic_bic_table = pd.DataFrame(aic_bic_results)

# Print results
print("\nAIC and BIC Results:")
print(aic_bic_table.sort_values(by="aic"))

# Save results in csv
# aic_bic_table.to_csv("Data/CopulaSelection_SE_aic_bic_results.csv", index=False)
# %%  ---- FITTING OF THE BEST COPULA ----     
#-- Fitting margins 
params_hs = expon.fit(Hs)
params_wl = gamma.fit(Wl) 

#-- Fitting the best copula family 
copFLOOD_T = pv.Bicop( pv.BicopFamily.gumbel, rotation = 90)
copFLOOD_T.fit(data = FLOOD_u)

print('Best fitted Copula on data (based on AIC test)')
print(copFLOOD_T)

#-- sim best theoretical
FLOOD_Tsim = copFLOOD_T.simulate(n = 10**6)
FLOOD_Tsim[:,0]  = FLOOD_Tsim[:,0]
FLOOD_Tsim[:,1]  = FLOOD_Tsim[:,1]

# Apply the transformation to the simulated data

FLOODfp_Tsim =  expon.ppf(FLOOD_Tsim[:,0], *params_hs)
FLOODfv_Tsim =  gamma.ppf(FLOOD_Tsim[:,1], *params_wl)

# -- values from theoretical copulas in [0,1] domain
FLOODfp_Tsim_domain  = rankdata(FLOODfp_Tsim)/(len(FLOODfp_Tsim)+1)
FLOODfv_Tsim_domain  = rankdata(FLOODfv_Tsim)/(len(FLOODfv_Tsim)+1)

# check transformation
fig   = plt.figure( figsize = (4,4))
gs    = gridspec.GridSpec( ncols = 1, nrows = 1, figure = fig)

ax11   = fig.add_subplot( gs[0, 0] )
ax11.scatter(FLOOD_Tsim[:,0],FLOODfp_Tsim_domain, s=20, alpha = 0.5, marker="o"
             , c = '0.8', edgecolor='0.3')
ax11.grid()

print(copFLOOD_T)

# %% Inputs
ny  = 36.48
Nt1 = len(Hs)
lambda1 = Nt1 / ny

WL_thresholds = np.linspace(min(FLOODfv_Tsim), max(FLOODfv_Tsim), 1000)
Hs_thresholds = np.linspace(min(FLOODfp_Tsim), max(FLOODfp_Tsim), 1000)

# Generate grid of thresholds
X, Y = np.meshgrid(WL_thresholds, Hs_thresholds)
u1 = gamma.cdf(X, *params_wl) 
u2 = expon.cdf(Y, *params_hs)

# Calculations
copula_cdf = copFLOOD_T.cdf(np.column_stack((u1.ravel(), u2.ravel()))).reshape(X.shape)

# Calculate OR and AND scenario probabilities
prob_or  = (1 - copula_cdf)*lambda1
prob_and = (1 - u1 - u2 + copula_cdf)*lambda1

# Return periods
Z_or = np.where(prob_or > 0,(1 / (prob_or)), float("inf"))
Z_and = np.where(prob_and > 0, (1 / (prob_and)), float("inf"))

# Plotting
fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
ftsiz = 12

# OR Scenario Plot
ax11 = fig.add_subplot(gs[0, 0])
contour_or = ax11.contour(X, Y, Z_or, levels=[1, 10, 50, 100, 500, 1000, 10000], linewidths=1, colors='red')
ax11.scatter(FLOODfv_Tsim,FLOODfp_Tsim, s=20, alpha=0.9, marker="o", c='0.8', edgecolor="none", label="Sim")
ax11.scatter(Wl, Hs, s=35, marker="o", c="k", edgecolor='none', label='Obs.')
ax11.grid(color='.7', linestyle='dotted', zorder=-1)
ax11.set_xlabel(r'Water Level [m+ RH2000]', fontsize=ftsiz)
ax11.set_ylabel(r'Hm0 [m]', fontsize=ftsiz)
ax11.clabel(contour_or, inline=True, fontsize=ftsiz * 0.6, fmt="%d years")
ax11.set_title('OR Scenario')
ax11.set_xlim([min(FLOODfv_Tsim)-0.1, max(FLOODfv_Tsim)+0.1])
ax11.set_ylim([min(FLOODfp_Tsim)-0.1, max(FLOODfp_Tsim)+0.1])
ax11.tick_params(axis='both', labelsize=ftsiz)

# AND Scenario Plot
ax12 = fig.add_subplot(gs[0, 1])
contour_and = ax12.contour(X, Y, Z_and, levels=[1, 10, 50, 100, 500, 1000, 10000], linewidths=1, colors='red')
ax12.scatter(FLOODfv_Tsim,FLOODfp_Tsim, s=20, alpha=0.9, marker="o", c='0.8', edgecolor='none', label="Sim")
ax12.scatter(Wl, Hs, s=35, marker="o", c='k', edgecolor='none', label='Obs.')
ax12.grid(color='.7', linestyle='dotted', zorder=-1)
ax12.set_xlabel(r'Water Level [m+ RH2000]', fontsize=ftsiz)
ax12.set_ylabel(r'Hm0 [m] ', fontsize=ftsiz)
ax12.clabel(contour_and, inline=True, fontsize=ftsiz * 0.6, fmt="%d years")
ax12.set_title('AND Scenario')
ax12.set_xlim([min(FLOODfv_Tsim)-0.1, max(FLOODfv_Tsim)+0.1])
ax12.set_ylim([min(FLOODfp_Tsim)-0.1, max(FLOODfp_Tsim)+0.1])
ax12.tick_params(axis='both', labelsize=ftsiz)
ax12.legend()
# plt.tight_layout()
plt.show()
# %%
contour_data_and = []
for level, collection in zip(contour_and.levels, contour_and.collections):
    for path in collection.get_paths():
        vertices = path.vertices
        for v in vertices:
            contour_data_and.append([level, v[0], v[1]])  
            
contour_data_or = []
for level, collection in zip(contour_or.levels, contour_or.collections):
    for path in collection.get_paths():
        vertices = path.vertices
        for v in vertices:
            contour_data_or.append([level, v[0], v[1]])  # Store Level, X, Y

# Save to CSV
df_and = pd.DataFrame(contour_data_and, columns=["Level", "X", "Y"])
df_or  = pd.DataFrame(contour_data_or, columns=["Level", "X", "Y"])
# Uncomment to save
df_and.to_csv("Data/Falsterbo_05_South_BivRP_WD_AND_NW_RH2000.csv", index=False)
df_or.to_csv("Data/Falsterbo_06_South_BivRP_WD_OR_NW_RH2000.csv", index=False)

# %% Inputs probabilities
ny  = 36.48
Nt1 = len(Hs)
lambda1 = Nt1 / ny

WL_thresholds = np.linspace(min(FLOODfv_Tsim), max(FLOODfv_Tsim), 1000)
Hs_thresholds = np.linspace(min(FLOODfp_Tsim), max(FLOODfp_Tsim), 1000)

# Generate grid of thresholds
X, Y = np.meshgrid(WL_thresholds, Hs_thresholds)
u1 = gamma.cdf(X, *params_wl) 
u2 = expon.cdf(Y, *params_hs)

# Calculations
copula_cdf = copFLOOD_T.cdf(np.column_stack((u1.ravel(), u2.ravel()))).reshape(X.shape)

# Calculate OR and AND scenario probabilities
prob_or  = (1 - copula_cdf)
prob_and = (1 - u1 - u2 + copula_cdf)

# Plotting
fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
ftsiz = 12

contour_levels = [0.001, 0.01,0.1, 0.5,0.9,0.99]

# OR Scenario Plot
ax11 = fig.add_subplot(gs[0, 0])
contour_or = ax11.contour(X, Y, prob_or, levels=contour_levels, linewidths=1, colors='red')
ax11.scatter(FLOODfv_Tsim, FLOODfp_Tsim, s=20, alpha=0.9, marker="o", c='0.8', edgecolor="none", label="Sim")
ax11.scatter(Wl, Hs, s=35, marker="o", c="k", edgecolor='none', label='Obs.')
ax11.grid(color='.7', linestyle='dotted', zorder=-1)
ax11.set_xlabel(r'Water Level [m+ RH2000]', fontsize=ftsiz)
ax11.set_ylabel(r'Hm0 [m]', fontsize=ftsiz)
ax11.clabel(contour_or, inline=True, fontsize=ftsiz * 0.6, fmt='%1.3f')
ax11.set_title('OR Scenario')
ax11.set_xlim([min(FLOODfv_Tsim)-0.1, max(FLOODfv_Tsim)+0.1])
ax11.set_ylim([min(FLOODfp_Tsim)-0.1, max(FLOODfp_Tsim)+0.1])
ax11.tick_params(axis='both', labelsize=ftsiz)

# AND Scenario Plot
ax12 = fig.add_subplot(gs[0, 1])
contour_and = ax12.contour(X, Y, prob_and, levels=contour_levels, linewidths=1, colors='red')
ax12.scatter(FLOODfv_Tsim, FLOODfp_Tsim, s=20, alpha=0.9, marker="o", c='0.8', edgecolor='none', label="Sim")
ax12.scatter(Wl, Hs, s=35, marker="o", c='k', edgecolor='none', label='Obs.')
ax12.grid(color='.7', linestyle='dotted', zorder=-1)
ax12.set_xlabel(r'Water Level [m+ RH2000]', fontsize=ftsiz)
ax12.set_ylabel(r'Hm0 [m] ', fontsize=ftsiz)
ax12.clabel(contour_and, inline=True, fontsize=ftsiz * 0.6, fmt='%1.3f')
ax12.set_title('AND Scenario')
ax12.set_xlim([min(FLOODfv_Tsim)-0.1, max(FLOODfv_Tsim)+0.1])
ax12.set_ylim([min(FLOODfp_Tsim)-0.1, max(FLOODfp_Tsim)+0.1])
ax12.tick_params(axis='both', labelsize=ftsiz)
ax12.legend()
# plt.tight_layout()
plt.show()

# %% Empirical Copula for plotting (u and v are ordered)

# To plot isolines, we need to create a mesh over the unit square domain
X = np.linspace(0, 1, 100)  # Uniform values in the unit square
Y = np.linspace(0, 1, 100)
XX, YY = np.meshgrid(X, Y)  # Create 2D grid for X and Y
XX_YY  = np.column_stack([XX.ravel(), YY.ravel()])  # Flatten into 2D array for copula input

# Evaluate the copula CDF over the grid
cdf_copFLOOD_T_grid = copFLOOD_T.cdf(XX_YY).reshape(XX.shape)  # Reshape to match grid
print("Shape of cdf_copFLOOD_T_grid:", cdf_copFLOOD_T_grid.shape)

res = len(FLOOD_u)
q   = np.linspace(0.001, 0.999, res)
grid_x1, grid_x2 = np.meshgrid(q, q) # u and v values for the copula
xy = np.column_stack([grid_x1.ravel(), grid_x2.ravel()])

# Initialize empirical copula
cop_emp  = np.empty( [res,res] )

# Obtaining the copula value
index = 0;

for i in range(0, len(FLOOD_u)):
    for j in range(0, len(FLOOD_u)):
        
        mask_x1  = (FLOOD_u[:,0] <= grid_x1[i,j]) # IF IT IS TRUE RETURNS 1 ELSE RETURNS 0
        mask_x2  = (FLOOD_u[:,1] <= grid_x2[i,j]) # IF IT IS TRUE RETURNS 1 ELSE RETURNS 0
        mask_x12 = (1*mask_x1) + (1*mask_x2)     # TRUE+TRUE=2, TRUE+FALSE=1, FALSE+FALSE=0
        
        copulaValue  = sum(mask_x12==2)/ (len(FLOOD_u)+1) # ONLY TRUE+TRUE=2 IS USED
        cop_emp[i,j] = copulaValue  
        
# Plot comparison between theoretical and empirical copula
contour_levels = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.99]
fig   = plt.figure( figsize = (5,5))
gs    = gridspec.GridSpec( ncols = 1, nrows = 1, figure = fig)

ax11   = fig.add_subplot( gs[0, 0] )

# ---------- Copula Domain
ax11.scatter(FLOOD_Tsim[:,0],FLOOD_Tsim[:,1], s=20, alpha = 0.5, marker="o", 
             c = '0.8', edgecolor='0.3', label = "Sim")
ax11.scatter(FLOOD_u[:,0],FLOOD_u[:,1], s=20, marker="o", 
             c = 'k', edgecolor='k', label = 'Obs.')

contour   = ax11.contour(XX, YY, cdf_copFLOOD_T_grid, levels=contour_levels, 
                         linestyles='dotted', linewidths=2, colors = 'red', 
                         label = 'Theorical Cop.')

contour_e = ax11.contour(grid_x1, grid_x2, cop_emp, levels=contour_levels, 
                          linestyles='dotted', linewidths=2, colors = 'blue', 
                          label = ' Emp. Cop.')

ax11.grid(color = '.7', linestyle='dotted', zorder=-1)
ax11.set_xlabel(r'$u$', fontsize = 12)
ax11.set_ylabel(r'$v$', fontsize = 12)

# Add contour labels with CDF values
ax11.clabel(contour, fmt='%1.2f', inline=True, fontsize=12)
ax11.clabel(contour_e, fmt='%1.2f', inline=True, fontsize=12)
ax11.legend(loc = 'lower left', ncol = 1, fontsize = 12)

# %% Empirical Copula for plotting (u and v are ordered)

# To plot isolines, we need to create a mesh over the unit square domain
X = np.linspace(0, 1, 100)  # Uniform values in the unit square
Y = np.linspace(0, 1, 100)
XX, YY = np.meshgrid(X, Y)  # Create 2D grid for X and Y
XX_YY  = np.column_stack([XX.ravel(), YY.ravel()])  # Flatten into 2D array for copula input

# Evaluate the copula CDF over the grid
cdf_copFLOOD_T_grid = copFLOOD_T.cdf(XX_YY).reshape(XX.shape)  # Reshape to match grid
print("Shape of cdf_copFLOOD_T_grid:", cdf_copFLOOD_T_grid.shape)

res = len(FLOOD_u)
q   = np.linspace(0.001, 0.999, res)
grid_x1, grid_x2 = np.meshgrid(q, q) # u and v values for the copula
xy = np.column_stack([grid_x1.ravel(), grid_x2.ravel()])

# Initialize empirical copula
cop_emp  = np.empty( [res,res] )

# Obtaining the copula value
index = 0;

for i in range(0, len(FLOOD_u)):
    for j in range(0, len(FLOOD_u)):
        
        mask_x1  = (FLOOD_u[:,0] <= grid_x1[i,j]) # IF IT IS TRUE RETURNS 1 ELSE RETURNS 0
        mask_x2  = (FLOOD_u[:,1] <= grid_x2[i,j]) # IF IT IS TRUE RETURNS 1 ELSE RETURNS 0
        mask_x12 = (1*mask_x1) + (1*mask_x2)     # TRUE+TRUE=2, TRUE+FALSE=1, FALSE+FALSE=0
        
        copulaValue  = sum(mask_x12==2)/ (len(FLOOD_u)+1) # ONLY TRUE+TRUE=2 IS USED
        cop_emp[i,j] = copulaValue  
        

Prob_OR = (1 - cdf_copFLOOD_T_grid)
Prob_AND = (1 - XX - YY + cdf_copFLOOD_T_grid)

# Plot comparison between theoretical and empirical copula
# contour_levels = [0.001, 0.002, 0.01, 0.02, 0.1]
contour_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.99]

fig   = plt.figure( figsize = (10,6))
gs    = gridspec.GridSpec( ncols = 2, nrows = 1, figure = fig)

ax11   = fig.add_subplot( gs[0, 0] )

# ---------- Copula Domain
ax11.scatter(FLOOD_Tsim[:,0],FLOOD_Tsim[:,1], s=20, alpha = 0.5, marker="o", 
             c = '0.8', edgecolor='0.3', label = "Sim")
ax11.scatter(FLOOD_u[:,0],FLOOD_u[:,1], s=20, marker="o", 
             c = 'k', edgecolor='k', label = 'Obs.')

contour   = ax11.contour(XX, YY, Prob_OR, levels=contour_levels, 
                         linestyles='dotted', linewidths=2, colors = 'red', 
                         label = 'Threorical Cop.')

ax11.grid(color = '.7', linestyle='dotted', zorder=-1)
ax11.set_xlabel(r'$u$', fontsize = 12)
ax11.set_ylabel(r'$v$', fontsize = 12)
ax11.set_title('OR Scenario')

# Add contour labels with CDF values
ax11.clabel(contour, fmt='%1.2f', inline=True, fontsize=12)
ax11.clabel(contour_e, fmt='%1.2f', inline=True, fontsize=12)
ax11.legend(loc = 'lower left', ncol = 1, fontsize = 12)


# ---------- Copula Domain
ax12   = fig.add_subplot( gs[0,1] )

ax12.scatter(FLOOD_Tsim[:,0],FLOOD_Tsim[:,1], s=20, alpha = 0.5, marker="o", 
             c = '0.8', edgecolor='0.3', label = "Sim")
ax12.scatter(FLOOD_u[:,0],FLOOD_u[:,1], s=20, marker="o", 
             c = 'k', edgecolor='k', label = 'Obs.')

contour   = ax12.contour(XX, YY, Prob_AND, levels=contour_levels, 
                         linestyles='dotted', linewidths=2, colors = 'red', 
                         label = 'Threorical Cop.')

ax12.grid(color = '.7', linestyle='dotted', zorder=-1)
ax12.set_xlabel(r'$u$', fontsize = 12)
ax12.set_ylabel(r'$v$', fontsize = 12)

# Add contour labels with CDF values
ax12.clabel(contour, fmt='%1.2f', inline=True, fontsize=12)
ax12.clabel(contour_e, fmt='%1.2f', inline=True, fontsize=12)
ax12.legend(loc = 'lower left', ncol = 1, fontsize = 12)
ax12.set_title('AND Scenario')
