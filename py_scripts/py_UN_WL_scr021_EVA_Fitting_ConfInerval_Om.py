#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:54:27 2025

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

from extremalidx import extremalidx  # Import the function from the saved module
from runup import runup
from dispersion_index import dispersion_index

def ecdf(data):
    sorted_data = np.sort(data)
    cdf = np.zeros_like(sorted_data, dtype=float)
    for ii in range(len(sorted_data)):
        cdf[ii] = (ii + 1) / (len(sorted_data) + 1)
    return sorted_data, cdf

# %% Load data
Data = pd.read_csv('Data/Data_Falsterbo10958_YSTAD.csv', parse_dates=['Time'])
Data.set_index('Time', inplace=True)

# Total Water Level Calculation
MSL         = 14.3-(-0.07)*(2020-2000) # MSL in RH2000 (cm)
Data['SSH'] = (Data['SSH'] + MSL)/ 100

# %%  Fitting Distributions for WL
Var  = Data['SSH']

# POT 
Threshold  = np.quantile(Var, 0.998)
DeclusTime = 2

POT_var = get_extremes(Var, "POT", threshold=Threshold, 
                                r=f"{DeclusTime}d")

# Calculate Lambda
ny      = len(Data.index)/(24*365)
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
plt.ylim([Threshold-0.1, 2.5])
plt.xlabel('TR [years]', fontweight='bold')
plt.ylabel('Water Level [m + RH2000]', fontweight='bold')
plt.grid(True, which="both", linestyle='dotted', alpha=0.7)
plt.legend(loc='upper left', fontsize=7)
plt.title('Return Period for Water Level', fontsize=10)
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
# %% Bootstrap Sensitivity 
Var       = Data['SSH']  
Threshold = np.quantile(Var, 0.998)

# POT
POT_var = get_extremes(Var, "POT", threshold=Threshold, r="2d")  # Adjust as needed

max_iterations = 1000
iteration_counts = np.logspace(0, np.log10(max_iterations), num=int(np.log10(max_iterations) * 10), dtype=int)

shape_cv = []
scale_cv = []

# Bootstrap Loop
np.random.seed(42) 
for n in iteration_counts:
    shape_samples = []
    scale_samples = []

    for _ in range(n):
        # Bootstrap resample
        bootstrap_sample = np.random.choice(POT_var, size=len(POT_var), replace=True)
        
        # Fit GPD
        # params = genpareto.fit(bootstrap_sample - Threshold, floc=0, scale=Threshold)
        params = expon.fit(bootstrap_sample)
        shape_samples.append(params[0])  # Shape parameter
        scale_samples.append(params[1])  # Scale parameter

    # Calculate CV for shape and scale parameters
    shape_cv.append(np.std(shape_samples) / np.mean(shape_samples))
    scale_cv.append(np.std(scale_samples) / np.mean(scale_samples))

# Plot Results
plt.figure(figsize=(6, 6))

plt.subplot(2, 1, 1)
plt.plot(iteration_counts, shape_cv, linewidth = 2,  linestyle = '-',
         marker='none', label='Shape CV', color='black')
plt.xscale("log")
plt.xlabel("Number of iterations", fontsize=12)
plt.ylabel("CV", fontsize=12)
plt.title("Shape parameter", fontsize=14)
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend(fontsize=10)

plt.subplot(2, 1, 2)
plt.plot(iteration_counts, scale_cv, linewidth = 2,  linestyle = '-',
         marker='none', label='Scale CV', color='black')
plt.xscale("log")
plt.xlabel("Number of iterations", fontsize=12)
plt.ylabel("CV", fontsize=12)
plt.title("Scale parameter", fontsize=14)
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()
# %% Return period and Confidence interval 
Var  = Data['SSH']

# POT 
Threshold  = np.quantile(Var, 0.998)
DeclusTime = 2

POT_var = get_extremes(Var, "POT", threshold=Threshold, 
                                r=f"{DeclusTime}d")

# Calculate Lambda
ny      = len(Var)/(365.25*24)
Nt      = len(POT_var)
lambda_ = Nt / ny

# Fit Distributions
# gpd_params = genpareto.fit(POT_var - Threshold, floc=0, scale=Threshold)  # GPD fit above threshold
# gpd_params = gpd_params_hs[:1] + (Threshold,) + gpd_params_hs[2:]
exp_params = expon.fit(POT_var)

x, f = ecdf(POT_var)
f[f == 1] = 0.999999
rp = 1 / ((1 - f) * lambda_)

# Define return period range
Rp = np.concatenate([np.arange(round(rp.min(), 2) + 0.01, 2, 0.01), np.arange(2, 20001)])
YPlot = 1 - 1 / (Rp * lambda_)

# Compute return values
# RV_gpd = genpareto.ppf(YPlot, *gpd_params)
RV_gpd = expon.ppf(YPlot, *exp_params)

# Number of bootstrap iterations
num_bootstrap = 250

# Storage for bootstrap return values
bootstrap_rv = np.zeros((num_bootstrap, len(Rp)))

# Bootstrap loop
for i in range(num_bootstrap):
    bootstrap_sample = np.random.choice(POT_var, size=len(POT_var), replace=True)
    
    # Fit GPD to bootstrap sample
    # bootstrap_params = genpareto.fit(bootstrap_sample - Threshold, floc=0, scale=Threshold)
    # bootstrap_params = bootstrap_params[:1] + (Threshold,) + bootstrap_params[2:]  # Ensure the threshold
    bootstrap_params = expon.fit(bootstrap_sample)
    # Compute return values
    # bootstrap_rv[i, :] = genpareto.ppf(YPlot, *bootstrap_params)
    bootstrap_rv[i, :] = expon.ppf(YPlot, *bootstrap_params)

# Calculate confidence intervals (e.g., 95%)
lower_ci = np.percentile(bootstrap_rv, 2.5, axis=0)
upper_ci = np.percentile(bootstrap_rv, 97.5, axis=0)

# Plotting with Confidence Intervals
plt.figure(figsize=(6, 5))

plt.plot(Rp, RV_gpd, label='Exponential', color='red',linestyle = '-',
         linewidth=2)

plt.fill_between(Rp, lower_ci, upper_ci, color='blue', alpha=0.15,linestyle = '--',
                 label='95% CI')

plt.scatter(rp, x, label='Empirical Data', color='gray', edgecolor='k', 
            alpha=0.7, zorder=10, linewidth=0.5)

plt.xscale('log')
plt.xlim([min(rp) - 1, 1000])
plt.ylim([Threshold - 0.1, 2.0])
plt.xlabel('TR [years]', fontweight='bold')
plt.ylabel('Water Level [m + RH2000]', fontweight='bold')
plt.grid(True, which="both", linestyle='dotted', alpha=0.7)
plt.legend(loc='upper left', fontsize=7)
plt.title('Return Period with 95% CI for Water Level', fontsize=10)
plt.tight_layout()
plt.show()

# %%
Var  = Data['SSH']

# POT 
Threshold  = np.quantile(Var, 0.998)
DeclusTime = 2

model = EVA(Var)
model.get_extremes(method = "POT", threshold=Threshold,r=f"{DeclusTime}d")        
model.fit_model()

summary = model.get_summary(
    return_period=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 10000],
    alpha=0.95,
    n_samples=500,
)

print(summary)

model.plot_diagnostic(alpha=0.95)
