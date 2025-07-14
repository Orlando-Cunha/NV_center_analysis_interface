import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional
from analysis_utils import extract_roi, split_signal_reference, average_over_roi, plot_roi_on_image, plot_signal_reference, plot_normalized_signal
from analysis_results import RabiResult

# Constants
FIG_SIZE = (8, 8)
RECT_LINEWIDTH = 2
RECT_EDGE_COLOR = 'red'
RECT_FACE_COLOR = 'none'
PLOT1_COLOR_SIGNAL = 'r'
PLOT1_COLOR_REFERENCE = 'b'
PLOT2_FMT = '-o'


class RabiResult:
    def __init__(self, x, y, error, mean_signal, mean_reference, image=None, roi_bounds=None):
        self.x = x
        self.y = y
        self.error = error
        self.mean_signal = mean_signal
        self.mean_reference = mean_reference
        self.image = image
        self.roi_bounds = roi_bounds


def rabi_analyze_data(
    image: np.ndarray,
    data: np.ndarray,
    x_range: np.ndarray,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    num_averages: int,
    num_points: int
) -> RabiResult:
    """
    Analyze Rabi oscillation data for a selected region of interest (ROI).

    Parameters:
        image (np.ndarray): The reference image for plotting the ROI.
        data (np.ndarray): The full data array (images over time or frequency).
        x_range (np.ndarray): The x-axis values (Rabi times).
        x_min, x_max, y_min, y_max (int): ROI boundaries.
        num_averages (int): Number of averages in the experiment.
        num_points (int): Number of points in the experiment.

    Returns:
        RabiResult: Contains x values, normalized signal, and propagated error.
    """
    print(f"Running analysis on data with shape: {data.shape} and x_range: {x_range}")

    # Split data into signal and reference (alternating frames)
    signal, reference = split_signal_reference(data)
    # Extract ROI from each frame
    signal = extract_roi(signal, x_min, x_max, y_min, y_max)
    reference = extract_roi(reference, x_min, x_max, y_min, y_max)
    if signal.shape != reference.shape:
        raise ValueError("Signal and reference datasets have different shapes.")

    # Average over the ROI for each frame
    average_pixel_values_signal = average_over_roi(signal)
    average_pixel_values_reference = average_over_roi(reference)

    # Reshape to (num_points, num_averages) for further averaging
    average_pixel_values_signal = np.reshape(average_pixel_values_signal, (num_points, num_averages))
    average_pixel_values_reference = np.reshape(average_pixel_values_reference, (num_points, num_averages))

    # Calculate mean and standard deviation for each time point
    mean_signal = np.mean(average_pixel_values_signal, axis=1)
    mean_reference = np.mean(average_pixel_values_reference, axis=1)
    std_signal = np.std(average_pixel_values_signal, axis=1)
    std_reference = np.std(average_pixel_values_reference, axis=1)

    # Normalize the signal by the reference
    final = mean_signal / mean_reference
    # Propagate error for the normalized signal
    std_final = np.sqrt((std_signal / mean_signal) ** 2 + (std_reference / mean_reference) ** 2) * final

    if len(x_range) != len(final):
        raise ValueError(f"Length mismatch: x_range {len(x_range)} and final {len(final)}")

    # Return all relevant data for GUI plotting
    return RabiResult(
        x=x_range,
        y=final,
        error=std_final,
        mean_signal=mean_signal,
        mean_reference=mean_reference,
        image=image,
        roi_bounds=(x_min, x_max, y_min, y_max)
    )
