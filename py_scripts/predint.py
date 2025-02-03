# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:11:53 2025

@author: rrodriguesbend
"""
import numpy as np
from scipy.stats import t

# Define the predint function
def predint(model_func, fitresult, x, confidence=0.95, mode='functional', 
            simultaneous='off', residual_std=None):
    params, covariance = fitresult
    y_pred = model_func(x, *params)

    # Finite difference gradients
    gradients = np.zeros((len(x), len(params)))
    delta = 1e-5  # Small step for numerical derivatives
    for i, param in enumerate(params):
        params_step = params.copy()
        params_step[i] += delta
        gradients[:, i] = (model_func(x, *params_step) - y_pred) / delta

    # Variance of the mean prediction
    var_mean = np.einsum('ij,jk,ik->i', gradients, covariance, gradients)
    std_mean = np.sqrt(var_mean)

    # Adjust for observation bounds if needed
    if mode == 'observation':
        if residual_std is None:
            raise ValueError("residual_std is required for 'observation' bounds.")
        std_total = np.sqrt(var_mean + residual_std**2)
    else:
        std_total = std_mean

    # Critical value
    n = len(x)
    dof = len(x) - len(params)
    t_value = t.ppf(1 - (1 - confidence) / 2, df=dof)

    if simultaneous == 'on':
        hw_factor = np.sqrt(n) * t_value  # Working-Hotelling adjustment
    else:
        hw_factor = t_value

    # Compute bounds
    lower_bound = y_pred - hw_factor * std_total
    upper_bound = y_pred + hw_factor * std_total

    return lower_bound, upper_bound

