import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from typing import Tuple
from analysis_utils import extract_roi, split_signal_reference, average_over_roi, plot_roi_on_image, plot_signal_reference, plot_normalized_signal
from analysis_results import RamseyResult

# Constants
FIG_SIZE = (8, 8)
RECT_LINEWIDTH = 2
RECT_EDGE_COLOR = 'red'
RECT_FACE_COLOR = 'none'
PLOT1_COLOR_SIGNAL = 'r'
PLOT1_COLOR_REFERENCE = 'b'
PLOT2_FMT = '-o'


def ramsey_analyze_data(
    image: np.ndarray,
    data: np.ndarray,
    x_range: np.ndarray,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    num_averages: int,
    num_points: int
) -> RamseyResult:
    """
    Analyze Ramsey interferometry data for a selected region of interest (ROI).

    Parameters:
        image (np.ndarray): The reference image for plotting the ROI.
        data (np.ndarray): The full data array (images over time or frequency).
        x_range (np.ndarray): The x-axis values (Ramsey times).
        x_min, x_max, y_min, y_max (int): ROI boundaries.
        num_averages (int): Number of averages in the experiment.
        num_points (int): Number of points in the experiment.

    Returns:
        RamseyResult: Contains x values and normalized signal.
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

    # Reshape to (num_points, num_averages) for further averaging
    average_pixel_values_signal = np.reshape(average_pixel_values_signal, (num_points, num_averages))
    average_pixel_values_reference = np.reshape(average_pixel_values_reference, (num_points, num_averages))

    # Average across all averages to get the final trace
    average_pixel_values_signal = np.mean(average_pixel_values_signal, axis=(1))
    average_pixel_values_reference = np.mean(average_pixel_values_reference, axis=(1))
    # Normalize the signal by the reference
    final = average_pixel_values_signal / average_pixel_values_reference
    if len(x_range) != len(final):
        print(f"Length mismatch: x_range {len(x_range)} and final {len(final)}")
        return
    # Create a figure for visualization
    fig = plt.figure(figsize=FIG_SIZE)
    # Plot the reference image and highlight the ROI
    image_plot = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=1)
    plot_roi_on_image(image, x_min, x_max, y_min, y_max, ax=image_plot)
    # Plot mean signal and reference data
    plot1 = plt.subplot2grid((3, 3), (0, 1), colspan=3)
    plot_signal_reference(x_range, average_pixel_values_signal, average_pixel_values_reference, xlabel='Relaxation time (ns)', ylabel='Intensity', title='Averaged Signal', ax=plot1)
    # Plot normalized signal
    plot2 = plt.subplot2grid((3, 3), (1, 1), colspan=3, rowspan=2)
    plot_normalized_signal(x_range, final, xlabel='Relaxation time (ns)', ylabel='Intensity', title='Ramsey Interferometry', ax=plot2)
    plt.tight_layout()
    plt.show()
    # Return x values and normalized signal as RamseyResult
    return RamseyResult(x=x_range, y=final)
