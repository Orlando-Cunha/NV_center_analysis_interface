import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename
from scipy.signal import detrend

# GUI to select file
tk.Tk().withdraw()
fn = askopenfilename(title="Select CSV file with Rabi oscillation data")
data = pd.read_csv(fn)

# Reverse data order
data_reordered = data.iloc[::-1].reset_index(drop=True)
tau_RAM = data_reordered['X'] * 1e-9
fluorescence = data_reordered['Y']

# Normalize
fluorescence_min = np.min(fluorescence)
fluorescence_max = np.max(fluorescence)
normalized_fluorescence = (fluorescence - fluorescence_min) / (fluorescence_max - fluorescence_min)

# Smoothing
window_length = 11
poly_order = 3
smoothed_fluorescence = savgol_filter(normalized_fluorescence, window_length, poly_order)
#smoothed_fluorescence = detrend(smoothed_fluorescence)
smoothed_fluorescence =normalized_fluorescence
# Save smoothed data
updated_data = pd.DataFrame({'X': tau_RAM, 'Y': smoothed_fluorescence})

from scipy.interpolate import interp1d

# Interpolate to uniform time spacing
uniform_time = np.linspace(tau_RAM.min(), tau_RAM.max(), len(tau_RAM))
interp_func = interp1d(tau_RAM, smoothed_fluorescence, kind='cubic')
uniform_signal = interp_func(uniform_time)

# Compute FFT
fft_result = np.fft.fft(uniform_signal)
fft_freq = np.fft.fftfreq(len(uniform_time), d=(uniform_time[1] - uniform_time[0]))
fft_amplitude = np.abs(fft_result)

# Filter out negative frequencies
positive_freqs = fft_freq > 0
fft_freq_mhz = fft_freq[positive_freqs] * 1e-6
fft_amplitude = fft_amplitude[positive_freqs]

# Plot FFT
plt.figure(figsize=(10, 4))
plt.plot(fft_freq_mhz, fft_amplitude, color='darkgreen')
plt.title('FFT of Ramsey Signal')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()
def double_exp_decay_cosine(t, A1, T1, f1, phi1, A2, T2, f2, phi2, y0):
    return A1 * np.exp(-t / T1)**2 * np.cos(2 * np.pi * f1 * t + phi1) + \
           A2 * np.exp(-t / T2)**2 * np.cos(2 * np.pi * f2 * t + phi2) + y0

def exp_decay_single_cos(t, A, B, T, beta, f, phi, y0):
    return B * t + A * np.exp(-t / T - beta * t**2) * np.cos(2 * np.pi * f * t + phi) + y0
# Fit: 2 Cos
try:
    popt_2cos = curve_fit(double_exp_decay_cosine, tau_RAM, smoothed_fluorescence,
                          p0=[0.5, 80e-9, 37e6, 0, 0.3, 80e-9, 35e6, 0, 0.3], maxfev=10000)[0]
    print("2 Cos Fit:", popt_2cos)
except Exception as e:
    popt_2cos = None
    print("2 Cos Fit failed:", e)
# Fit: 1 Cos with Gaussian damping
try:
    popt_1cos = curve_fit(exp_decay_single_cos, tau_RAM, smoothed_fluorescence,
                          p0=[0.5, 0.01, 80e-9, 1e9, 37e6, 0, 0.3], maxfev=10000)[0]
    print("1 Cos Fit with damping:", popt_1cos)
except Exception as e:
    popt_1cos = None
    print("1 Cos Fit with damping failed:", e)
plt.figure(figsize=(10, 4))


# 2 Cos
plt.subplot(1, 2, 1)
plt.plot(tau_RAM, smoothed_fluorescence, 'o', label="Data", color='black', alpha=0.7)
if popt_2cos is not None:
    plt.plot(tau_RAM, double_exp_decay_cosine(tau_RAM, *popt_2cos), label="Fit", color="red")
    plt.text(0.5, 0.9, f"T2 = {popt_2cos[5]:.2e}", transform=plt.gca().transAxes, ha='center', color='blue')
plt.legend(); plt.title("2 cosine Fit")

# 1 Cos w/ Damping
plt.subplot(1, 2, 2)
plt.plot(tau_RAM, smoothed_fluorescence, 'o', label="Data", color='black', alpha=0.7)
if popt_1cos is not None:
    plt.plot(tau_RAM, exp_decay_single_cos(tau_RAM, *popt_1cos), label="Fit", color="red")
    plt.text(0.5, 0.9, f"T2 = {popt_1cos[2]:.2e}", transform=plt.gca().transAxes, ha='center', color='blue')
plt.legend(); plt.title("1 cosine w/ damping")


plt.tight_layout()
plt.show()
# --- Calculate SNR for 1-cos fit ---
if popt_1cos is not None:
    fit_1cos = exp_decay_single_cos(tau_RAM, *popt_1cos)
    residuals_1cos = smoothed_fluorescence - fit_1cos
    noise_std_1cos = np.std(residuals_1cos)
    signal_amp_1cos = np.max(fit_1cos) - np.min(fit_1cos)
    snr_1cos = signal_amp_1cos / noise_std_1cos
    print(f"SNR (1 cosine fit): {snr_1cos:.2f}")

# --- Calculate SNR for 2-cos fit ---
if popt_2cos is not None:
    fit_2cos = double_exp_decay_cosine(tau_RAM, *popt_2cos)
    residuals_2cos = smoothed_fluorescence - fit_2cos
    noise_std_2cos = np.std(residuals_2cos)
    signal_amp_2cos = np.max(fit_2cos) - np.min(fit_2cos)
    snr_2cos = signal_amp_2cos / noise_std_2cos
    print(f"SNR (2 cosine fit): {snr_2cos:.2f}")
