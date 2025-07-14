import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import savgol_filter, detrend
from scipy.optimize import curve_fit
from find_peaks_func import find_valleys
from pybaselines import Baseline
import numpy as np
import pandas as pd
from typing import Callable, Tuple, Any

# Constants
DEFAULT_WINDOW_LENGTH = 11
DEFAULT_POLYORDER = 3
DEFAULT_INITIAL_PEAK_WIDTH = 0.005
DEFAULT_BASELINE_SMOOTH_HALF_WINDOW = 50
DEFAULT_BASELINE_STD_DISTRIBUTION_PARAM = 8
DEFAULT_MAXFEV = 100000

# Define Lorentzian Function
def lorentzian_2(x: np.ndarray, A: float, C: float, W: float) -> np.ndarray:
    """
    Define a single Lorentzian peak.
    Parameters:
        x (np.ndarray): x-axis values (e.g., frequency)
        A (float): Amplitude
        C (float): Center
        W (float): Half-width at half-maximum (HWHM)
    Returns:
        np.ndarray: Lorentzian peak values
    """
    return A / ((1 + ((x - C) / W) ** 2))

# Smoothing Function
def smooth_function(y_data: np.ndarray, window_length: int = DEFAULT_WINDOW_LENGTH, polyorder: int = DEFAULT_POLYORDER) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to smooth data.
    Parameters:
        y_data (np.ndarray): Input data to smooth
        window_length (int): Length of the filter window
        polyorder (int): Order of the polynomial used to fit the samples
    Returns:
        np.ndarray: Smoothed data
    """
    return savgol_filter(y_data / max(y_data), window_length=window_length, polyorder=polyorder)

# Baseline Correction
def baseline_correction_arpls(y_data: np.ndarray, x_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct baseline using ARPLS (Asymmetric Reweighted Penalized Least Squares).
    Parameters:
        y_data (np.ndarray): Input data
        x_data (np.ndarray): x-axis values
    Returns:
        tuple: (corrected_y, baseline)
    """
    baseline_fitter = Baseline(x_data=x_data)
    baseline, params = baseline_fitter.std_distribution(y_data, DEFAULT_BASELINE_STD_DISTRIBUTION_PARAM, smooth_half_window=DEFAULT_BASELINE_SMOOTH_HALF_WINDOW)
    corrected_y = y_data - baseline
    # Optionally detrend the corrected data
    # corrected_y = detrend(corrected_y)
    return corrected_y, baseline

# Peak Detection
def find_peaks(num_peaks: int, threshold: float, y_data: np.ndarray, x_data: np.ndarray, plot: bool = False) -> pd.Index:
    """
    Find valleys corresponding to peaks in the ODMR spectrum.
    Parameters:
        num_peaks (int): Number of peaks to find
        threshold (float): Initial threshold for peak finding
        y_data (np.ndarray): Data to search for peaks
        x_data (np.ndarray): x-axis values
        plot (bool): Whether to plot the detected peaks
    Returns:
        pd.Index: Indices of detected valleys
    """
    valley_indices = pd.Index([], dtype='int64')
    # Incrementally increase threshold until enough valleys are found
    while len(valley_indices) < num_peaks and threshold < 1:
        valley_indices = find_valleys(x_data, y_data, threshold, False)
        threshold += 0.0005
    valley_indices = find_valleys(x_data, y_data, threshold, False)
    return valley_indices

# Generate Lorentzian Function
def generate_lorentzian_function(num_peaks: int) -> Callable[[np.ndarray, Any], np.ndarray]:
    """
    Create a multi-peak Lorentzian function for fitting.
    Parameters:
        num_peaks (int): Number of Lorentzian peaks
    Returns:
        function: A function that computes the sum of num_peaks Lorentzian peaks
    """
    def fit_function(x: np.ndarray, *params: float) -> np.ndarray:
        if len(params) != num_peaks * 3:
            raise ValueError(f"Expected {num_peaks * 3} parameters, got {len(params)}.")
        result = 0
        # Sum each Lorentzian peak
        for i in range(0, len(params), 3):
            result += lorentzian_2(x, *params[i:i + 3])
        return result
    return fit_function

# Fitting Function
def fit_odmr_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    corrected_y: np.ndarray,
    valley_indices: pd.Index,
    baseline_offset: np.ndarray,
    plot: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the ODMR spectrum with multiple Lorentzian peaks.
    Parameters:
        x_data (np.ndarray): x-axis values
        y_data (np.ndarray): original y data
        corrected_y (np.ndarray): baseline-corrected y data
        valley_indices (pd.Index): indices of detected valleys
        baseline_offset (np.ndarray): baseline values
        plot (bool): Whether to plot the fit and data
    Returns:
        tuple: (popt, pcov, yfit) - fit parameters, covariance, and fitted curve
    """
    num_peaks = len(valley_indices)
    fit_function = generate_lorentzian_function(num_peaks)
    initial_guesses = []
    # Prepare data for fitting
    intensities = corrected_y + 1
    inverted_intensities = -intensities - min(-intensities)
    # Initial guesses for each peak: amplitude, center, width
    for i in range(num_peaks):
        initial_guesses.extend([intensities[valley_indices[i]], x_data[valley_indices[i]], DEFAULT_INITIAL_PEAK_WIDTH])
    # Perform curve fitting
    popt, pcov = curve_fit(fit_function, x_data, inverted_intensities, p0=initial_guesses, maxfev=DEFAULT_MAXFEV)
    yfit = fit_function(x_data, *popt)
    if plot:
        # Plot each individual peak
        for i in range(num_peaks):
            yfit_i = lorentzian_2(x_data, *popt[i * 3:(i + 1) * 3])
            plt.plot(x_data, yfit_i, linestyle="-.", linewidth=1, label=f'Peak {i + 1}')
        # Plot data and combined fit
        plt.plot(x_data, inverted_intensities, color="grey", linestyle="-", linewidth=2, label='Data')
        plt.plot(x_data, yfit, color="blue", linestyle="--", label='Combined Fit')
        final_data = -y_data / max(y_data) + baseline_offset
        plt.plot(x_data, final_data - min(final_data), label='Original Data')
        plt.legend()
        plt.show()
    return popt, pcov, yfit

# ODMR Fitting Function
def process_odmr_fit(
    x_data: np.ndarray,
    y_data: np.ndarray,
    num_peaks: int = 4,
    threshold: float = 0.974,
    plot: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline for ODMR data fitting.
    Parameters:
        x_data (np.ndarray): 1D array of frequencies.
        y_data (np.ndarray): 1D array of intensities.
        num_peaks (int): Number of Lorentzian peaks to fit.
        threshold (float): Initial threshold for peak finding.
        plot (bool): Whether to plot the fit and data.
    Returns:
        tuple: (popt, yfit) - fit parameters and fitted curve values.
    """
    # Smooth the data
    smoothed_y = smooth_function(y_data)
    # Find peaks
    valley_indices = find_peaks(num_peaks, threshold, smoothed_y, x_data, plot=False)
    # Correct baseline
    corrected_y, baseline_offset = baseline_correction_arpls(smoothed_y, x_data)
    # Fit data
    popt, pcov, yfit = fit_odmr_data(x_data, y_data, corrected_y, valley_indices, baseline_offset, plot=plot)
    return popt, yfit
