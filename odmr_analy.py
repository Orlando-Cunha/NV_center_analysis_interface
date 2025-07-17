import numpy as np
import matplotlib.pyplot as plt
import scipy
from typing import Tuple, Any
from analysis_utils import extract_roi, average_over_roi, plot_roi_on_image
from analysis_results import ODMRResult

# Constants
IMAGE_SIZE = 512
FILTER_ORDER = 3
CUTOFF_FREQ = 0.08
ROI_LEVELS = [-0.1, 0.1]
COLORBAR_LABEL = 'Counts'
COLORBAR_SHRINK = 0.5


def odmr_analyze_data(
    image: np.ndarray,
    data: np.ndarray,
    x_range: np.ndarray,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    num_averages: int,
    num_points: int
) -> ODMRResult:
    """
    Analyze ODMR (Optically Detected Magnetic Resonance) data for a selected region of interest (ROI).

    Parameters:
        image (np.ndarray): The reference image for plotting the ROI.
        data (np.ndarray): The full data array (images over time or frequency).
        x_range (np.ndarray): The x-axis values (e.g., frequency).
        x_min, x_max, y_min, y_max (int): ROI boundaries.
        num_averages (int): Number of averages in the experiment.
        num_points (int): Number of points in the experiment.

    Returns:
        ODMRResult: Contains x values, normalized ODMR trace, and duplicate for compatibility.
    """
    # Convert input data to numpy array for safety
    data_f = np.array(data)
    # Prepare array to store averaged ROI values for each frame
    ROI_avg_x = np.zeros((1, len(data_f)))
    # Remove the first 3*num_points frames (experiment-specific)
    data_f = data_f[3*num_points:]
    # Define ROI boundaries
    trace_x1 = int(x_min)
    trace_x2 = int(x_max)
    trace_y1 = int(y_min)
    trace_y2 = int(y_max)
    print(len(data_f))
    # Loop through each frame and average the ROI
    for i in range(len(data_f)):
        ROI = extract_roi(np.expand_dims(data_f[i], axis=0), trace_x1, trace_x2, trace_y1, trace_y2)[0]
        #print(ROI)
        ROI_avg_x[0, i] = np.mean(ROI)
    #print(ROI_avg_x)
    #print(np.size(ROI_avg_x))
    # Reshape to (num_averages, num_points) for further averaging
    odmr = np.reshape(ROI_avg_x, (num_averages, num_points))
    #print(np.size(odmr))
    # Average across all averages to get the final trace
    odmr_trace = np.mean(odmr, axis=0)
    # Normalize the trace
    odmr_trace = odmr_trace / np.max(odmr_trace)
    # Create a mask for the ROI for plotting
    high = np.zeros([IMAGE_SIZE, IMAGE_SIZE])
    high[trace_x1:trace_x2, trace_y1:trace_y2] = 1
    # Apply a Butterworth filter to smooth the trace
    B, A = scipy.signal.butter(FILTER_ORDER, CUTOFF_FREQ, output='ba')
    fltflt = scipy.signal.filtfilt(B, A, odmr_trace, method='gust')
    plot = image
    np.flip(plot, 1)
    # Remove or comment out all plt.show() and figure creation
    # plt.show()  # <-- REMOVE THIS LINE
    # Return x values, normalized trace, and duplicate for compatibility as ODMRResult
    return ODMRResult(
        x=x_range,
        y=odmr_trace,
        y2=odmr_trace,
        image=image,
        x_min=trace_x1,
        x_max=trace_x2,
        y_min=trace_y1,
        y_max=trace_y2,
        filtered_y=fltflt
    )