import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for thread safety
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from typing import Tuple
from analysis_utils import extract_roi, split_signal_reference, average_over_roi, reshape_and_average, plot_roi_on_image, plot_signal_reference, plot_normalized_signal
from analysis_results import T1Result

# Constants
RESHAPE_SIGNAL = (21, 20)
RECT_LINEWIDTH = 2
RECT_EDGE_COLOR = 'red'
RECT_FACE_COLOR = 'none'
FIG_SIZE = (8, 8)
PLOT1_COLOR_SIGNAL = 'r'
PLOT1_COLOR_REFERENCE = 'b'
PLOT2_FMT = '-o'


def t1_analyze_data(
    image: np.ndarray,
    data: np.ndarray,
    x_range: np.ndarray,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int
) -> T1Result:
    """
    Analyze T1 relaxation data for a selected region of interest (ROI).

    Parameters:
        image (np.ndarray): The reference image for plotting the ROI.
        data (np.ndarray): The full data array (images over time or frequency).
        x_range (np.ndarray): The x-axis values (relaxation times).
        x_min, x_max, y_min, y_max (int): ROI boundaries.

    Returns:
        T1Result: Contains x values, normalized signal, and standard deviation.
    """
    print(f"Running analysis on data with shape: {data.shape} and x_range: {x_range}")
    # Split data into signal and reference (alternating frames)
    signal, reference = split_signal_reference(data)
    # Extract ROI from each frame
    signal = extract_roi(signal, x_min, x_max, y_min, y_max)
    reference = extract_roi(reference, x_min, x_max, y_min, y_max)
    if signal.shape != reference.shape:
        print("Datasets have different shapes.")
        return
    # Average over the ROI for each frame
    average_pixel_values_signal = average_over_roi(signal)
    average_pixel_values_reference = average_over_roi(reference)
    # Reshape to (21, 20) for further averaging (experiment-specific)
    mean_signal, std_signal = reshape_and_average(average_pixel_values_signal, RESHAPE_SIGNAL)
    mean_reference, std_reference = reshape_and_average(average_pixel_values_reference, RESHAPE_SIGNAL)
    # Compute normalized difference
    final = mean_reference - mean_signal
    final = final / max(final)
    # Propagate error for the normalized difference
    std_final = np.sqrt((std_signal / mean_signal) ** 2 + (std_reference / mean_reference) ** 2) * np.abs(final)
    if len(x_range) != len(final):
        print(f"Length mismatch: x_range {len(x_range)} and final {len(final)}")
        return
    # Reverse order so first image is highest value
    x_range = x_range[::-1]
    final = final[::-1]
    std_final = std_final[::-1]
    # Create a figure for visualization (offscreen)
    fig = plt.figure(figsize=FIG_SIZE)
    # Plot the reference image and highlight the ROI
    image_plot = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=1)
    plot_roi_on_image(image, x_min, x_max, y_min, y_max, ax=image_plot)
    # Plot raw signal and reference data
    plot1 = plt.subplot2grid((3, 3), (0, 1), colspan=3)
    plot_signal_reference(np.arange(len(mean_signal)), mean_signal, mean_reference, xlabel='Number of image', ylabel='Intensity', title='Raw data', ax=plot1)
    # Plot normalized signal with error bars
    plot2 = plt.subplot2grid((3, 3), (1, 1), colspan=3, rowspan=2)
    plot_normalized_signal(x_range, final, yerr=std_final, xlabel='Relaxation time (ms)', ylabel='Intensity', title='T1 measurements', ax=plot2)
    plt.tight_layout()
    plt.close(fig)  # Close the figure to free memory and avoid GUI issues
    # Return x values, normalized signal, and propagated error as T1Result, plus mean_signal and mean_reference for GUI
    result = T1Result(x=x_range, y=final, error=std_final)
    result.mean_signal = mean_signal
    result.mean_reference = mean_reference
    result.image = image
    result.roi_bounds = (x_min, x_max, y_min, y_max)
    return result