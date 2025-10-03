import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

def analyze_ppg_data(file_path):
    """
    Reads PPG data from a Teleplot JSON/CSV file, calculates the true sampling rate,
    finds peaks, computes HRV metrics, and displays a plot.
    """
    # 1. Read the data using pandas
    try:
        # Skip the header row and name the columns
        df = pd.read_csv(file_path, header=0, names=['timestamp', 'signal'])
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Convert columns to numeric types
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['signal'] = pd.to_numeric(df['signal'])

    # 2. Calculate the actual sampling rate (fs)
    # Get the average difference between timestamps
    avg_time_diff = df['timestamp'].diff().mean()
    fs = 1.0 / avg_time_diff
    print(f"--- Data Analysis ---")
    print(f"Successfully loaded {len(df)} data points.")
    print(f"Calculated Sampling Rate: {fs:.2f} Hz")

    # 3. Apply a properly configured filter for the low sampling rate
    raw_signal = df['signal'].values
    nyquist = 0.5 * fs
    low_cut = 0.5 / nyquist
    high_cut = (fs / 2 - 0.5) / nyquist # High cut must be less than Nyquist

    if high_cut <= low_cut:
        print("Warning: High cut frequency is too low. Using raw signal.")
        filtered_signal = raw_signal
    else:
        b, a = butter(2, [low_cut, high_cut], btype='band')
        filtered_signal = filtfilt(b, a, raw_signal)
        print(f"Applied bandpass filter between 0.5 Hz and {high_cut*nyquist:.2f} Hz.")

    # 4. Find peaks with parameters adapted for ~12.5 Hz
    # Min distance: Based on a max heart rate of 180 BPM
    # (60 sec / 180 beats) * 12.5 samples/sec = ~4.16 samples
    min_distance_samples = (60.0 / 180.0) * fs

    # Prominence: 25% of the signal's dynamic range
    prominence = (np.max(filtered_signal) - np.min(filtered_signal)) * 0.25

    peaks, _ = find_peaks(
        filtered_signal,
        distance=min_distance_samples,
        prominence=prominence,
        width=(1, 10) # 1 sample min, 10 samples max (~800ms)
    )

    print(f"Found {len(peaks)} peaks.")

    if len(peaks) < 2:
        print("Not enough peaks found to calculate HRV.")
        return

    # 5. Calculate HRV Metrics
    # Get the timestamps of the detected peaks
    peak_times = df['timestamp'].iloc[peaks].values
    # Calculate the intervals between peaks in milliseconds
    nn_intervals = np.diff(peak_times) * 1000

    # Filter out physiologically improbable intervals
    nn_intervals = nn_intervals[(nn_intervals > 300) & (nn_intervals < 2000)]

    if len(nn_intervals) >= 2:
        avg_interval = np.mean(nn_intervals)
        heart_rate = 60000.0 / avg_interval
        sdnn = np.std(nn_intervals)
        rmssd = np.sqrt(np.mean(np.diff(nn_intervals)**2))

        print("\n--- HRV Results ---")
        print(f"Heart Rate: {heart_rate:.2f} BPM")
        print(f"SDNN:       {sdnn:.2f} ms")
        print(f"RMSSD:      {rmssd:.2f} ms")
    else:
        print("Not enough valid intervals to calculate HRV after filtering.")


    # 6. Plot the results for visual confirmation
    plt.figure(figsize=(15, 6))
    plt.plot(df['timestamp'], raw_signal, label='Raw Signal', alpha=0.5, color='gray')
    plt.plot(df['timestamp'], filtered_signal, label='Filtered Signal', color='orange')
    plt.plot(df['timestamp'].iloc[peaks], filtered_signal[peaks], 'rx', markersize=10, label='Detected Peaks')
    plt.title('PPG Signal Analysis')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


# --- Main execution ---
if __name__ == '__main__':
    # Replace with the actual name of your data file
    file_name = 'teleplot_2025-8-22_16-2.json'
    analyze_ppg_data(file_name)