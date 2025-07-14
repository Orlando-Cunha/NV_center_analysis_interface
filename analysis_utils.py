import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional

# ROI extraction utility
def extract_roi(data: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
    """
    Extract a region of interest (ROI) from a 3D data array.
    Args:
        data (np.ndarray): 3D array (frames, x, y)
        x_min, x_max, y_min, y_max (int): ROI boundaries
    Returns:
        np.ndarray: ROI-extracted data (frames, x, y)
    """
    return data[:, x_min:x_max, y_min:y_max]

# Signal/reference splitting utility
def split_signal_reference(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into signal and reference by alternating frames.
    Args:
        data (np.ndarray): 3D array (frames, x, y)
    Returns:
        Tuple[np.ndarray, np.ndarray]: (signal, reference)
    """
    signal = data[::2]
    reference = data[1::2]
    return signal, reference

# Averaging utility
def average_over_roi(frames: np.ndarray) -> np.ndarray:
    """
    Average over spatial dimensions for each frame.
    Args:
        frames (np.ndarray): 3D array (frames, x, y)
    Returns:
        np.ndarray: 1D array of mean values per frame
    """
    return np.mean(frames, axis=(1, 2))

# Reshape and mean for experiment-specific averaging
def reshape_and_average(data: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape data and compute mean and std along the second axis.
    Args:
        data (np.ndarray): 1D or 2D array
        shape (Tuple[int, int]): Target shape (e.g., (num_points, num_averages))
    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean, std) along axis=1
    """
    reshaped = np.reshape(data, shape)
    mean = np.mean(reshaped, axis=1)
    std = np.std(reshaped, axis=1)
    return mean, std

# Plotting utilities
def plot_roi_on_image(image: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot the ROI rectangle on an image.
    Args:
        image (np.ndarray): 2D image
        x_min, x_max, y_min, y_max (int): ROI boundaries
        ax (plt.Axes, optional): Axis to plot on
    Returns:
        plt.Axes: The axis with the ROI overlay
    """
    if ax is None:
        ax = plt.gca()
    ax.imshow(image, cmap='gray')
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    return ax

def plot_signal_reference(x: np.ndarray, signal: np.ndarray, reference: np.ndarray, xlabel: str = '', ylabel: str = '', title: str = '', ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot signal and reference traces.
    Args:
        x (np.ndarray): x-axis values
        signal (np.ndarray): Signal values
        reference (np.ndarray): Reference values
        xlabel, ylabel, title (str): Labels
        ax (plt.Axes, optional): Axis to plot on
    Returns:
        plt.Axes: The axis with the plot
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(x, signal, 'r', label='Signal')
    ax.plot(x, reference, 'b', label='Reference')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax

def plot_normalized_signal(x: np.ndarray, normalized: np.ndarray, yerr: Optional[np.ndarray] = None, xlabel: str = '', ylabel: str = '', title: str = '', ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot normalized signal with optional error bars.
    Args:
        x (np.ndarray): x-axis values
        normalized (np.ndarray): Normalized signal
        yerr (np.ndarray, optional): Error bars
        xlabel, ylabel, title (str): Labels
        ax (plt.Axes, optional): Axis to plot on
    Returns:
        plt.Axes: The axis with the plot
    """
    if ax is None:
        ax = plt.gca()
    if yerr is not None:
        ax.errorbar(x, normalized, yerr=yerr, fmt='-o', label='Normalized Signal')
    else:
        ax.plot(x, normalized, '-o', label='Normalized Signal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax 

def preprocess_odmr_data(raw_data: np.ndarray, num_averages: int, num_points: int) -> np.ndarray:
    """
    Reshape and average the raw ODMR data to produce a (num_points, 512, 512) array.
    """
    reshaped = raw_data.reshape(num_averages, num_points, 512, 512)
    preprocessed_data = reshaped.mean(axis=0)
    return preprocessed_data

def extract_roi_trace(preprocessed_data: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
    """
    Extract and average the ODMR trace from a specific ROI.
    """
    roi = preprocessed_data[:, y_min:y_max, x_min:x_max]
    odmr_trace = roi.mean(axis=(1,2))
    odmr_trace = odmr_trace / np.max(odmr_trace)
    return odmr_trace