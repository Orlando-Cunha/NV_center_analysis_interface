from findpeaks import findpeaks
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import io
import pandas as pd
from typing import Any, Tuple, List

# Constants
DEFAULT_THRESHOLD = 0.5  # Example threshold value, adjust as needed
DEFAULT_WINDOW_SIZE = 5  # Example window size, adjust as needed


'''
This module provides a function to find valleys (local minima) in ODMR protocol data after averaging.
The main function, find_valleys, uses the findpeaks library to detect valleys below a given threshold.
'''

def find_peaks(signal: np.ndarray, threshold: float = DEFAULT_THRESHOLD, window_size: int = DEFAULT_WINDOW_SIZE) -> List[int]:
    """
    Find peaks in a 1D signal array using a simple threshold and window method.
    Args:
        signal (np.ndarray): The input signal array.
        threshold (float): Minimum value to qualify as a peak.
        window_size (int): Number of points to consider on each side for local maximum.
    Returns:
        List[int]: Indices of detected peaks.
    """
    peaks = []
    for i in range(window_size, len(signal) - window_size):
        window = signal[i - window_size:i + window_size + 1]
        if signal[i] == np.max(window) and signal[i] > threshold:
            peaks.append(i)
    return peaks


def find_valleys(
    x_data: np.ndarray,
    y_data: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    plot: bool = True
) -> pd.Index:
    """
    Find valleys (local minima) in the provided y_data using the findpeaks library.

    Parameters:
        x_data (np.ndarray): The x-axis data (e.g., frequency or time).
        y_data (np.ndarray): The y-axis data (e.g., intensity).
        threshold (float): Only valleys with y-values below this threshold are considered.
        plot (bool): If True, plot the data and highlight the detected valleys.

    Returns:
        filtered_indices (pd.Index): Indices of the detected valleys below the threshold.
    """
    # Suppress any output from findpeaks
    with contextlib.redirect_stdout(io.StringIO()):
        fp = findpeaks(method="peakdetect", whitelist=['valley'], lookahead=12, scale=True, togray=True, denoise=True)
        results: dict[str, Any] = fp.fit(y_data)

    # Extract the DataFrame with peak/valley information
    keys = list(results.keys())
    values = list(results.values())
    df = values[keys.index('df')]

    # Filter valley values based on threshold
    filtered_df = df[(df['valley'] == True) & (df['y'] < threshold)]
    filtered_indices = filtered_df.index
    x_filtered = x_data[filtered_indices]

    if plot:
        # Plot the original data and highlight the detected valleys
        plt.figure()
        plt.plot(x_data, y_data, label='Original Data')
        plt.scatter(x_filtered, filtered_df['y'], color='red', label='Filtered Valleys')
        plt.legend()
        plt.show()

    return filtered_indices
