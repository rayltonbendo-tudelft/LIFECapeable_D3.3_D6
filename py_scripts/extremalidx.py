# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:07:37 2025

@author: rrodriguesbend
"""

import numpy as np

def extremalidx(data):
    """
    Extremal Index measuring the dependence of data

    Parameters:
    data : numpy.ndarray
        A two-column matrix with sampled times in the first column
        and values in the second column. Sampling frequency must be 1Hz.

    Returns:
    ei : float
        Extremal index, a measure of independence.
    Tc : float
        Minimum distance between clusters (declustering time).
    """
    # Ensure data is a numpy array
    data = np.array(data)

    # Reshape data if only times are given
    if data.ndim == 1:
        data = data[:, None]

    ei = None
    Tc = None

    if data.size == 0:
        return ei, Tc
    elif data.size == 1:
        return 1, Tc

    # Calculate interexceedance times
    Ti = np.diff(np.sort(data[:, 0]))
    ei = _lcl_extrml_idx(Ti)

    # Calculate declustering time if needed
    if ei is not None and ei < 1:
        N = len(Ti) + 1
        C = int(np.floor(N * ei)) + 1
        sTi = np.sort(Ti)[::-1]  # Sort in descending order
        Tc = sTi[min(C, N - 1) - 1]  # Declustering time
    else:
        Tc = np.min(Ti)

    return ei, Tc

def _lcl_extrml_idx(Ti):
    """
    Local function to compute the extremal index based on interexceedance times.

    Parameters:
    Ti : numpy.ndarray
        Interexceedance times.

    Returns:
    ei : float
        Extremal index.
    """
    Tmax = np.max(Ti)

    if Tmax <= 1:
        return 0
    elif Tmax <= 2:
        return min(1, 2 * np.mean(Ti) ** 2 / np.mean(Ti ** 2))
    else:
        return min(1, 2 * np.mean(Ti - 1) ** 2 / np.mean((Ti - 1) * (Ti - 2)))

# Example usage:
# data = np.array([[1, 5], [2, 3], [3, 8], [5, 6]])
# ei, Tc = extremalidx(data)
# print("Extremal Index:", ei)
# print("Declustering Time:", Tc)
