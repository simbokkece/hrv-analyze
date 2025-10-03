import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from scipy.signal import find_peaks

# --- Configuration ---
# Serial port settings - adjust if your device uses a different configuration
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 460800 # A common baud rate, change if needed
# Plotting settings
DATA_WINDOW_SIZE = 300 # Number of data points to display in the moving window
# Peak detection settings - these may need tuning based on your signal quality
PEAK_MIN_HEIGHT = 100000 # Minimum signal value to be considered a peak
PEAK_MIN_DISTANCE = 15 # Minimum horizontal distance (in data points) between peaks

# --- Global Variables ---
# Using deque for efficient appending and popping from both ends
timestamps = deque(maxlen=DATA_WINDOW_SIZE)
ppg_signals = deque(maxlen=DATA_WINDOW_SIZE)
# Store peak times to calculate HRV
peak_times = []
latest_hrv = {"sdnn": 0.0, "rmssd": 0.0}

# --- Serial Connection Setup ---
try:
    # Initialize serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    # Allow time for the serial port to initialize
    time.sleep(2)
    print(f"Successfully connected to serial port {SERIAL_PORT} at {BAUD_RATE} baud.")
except serial.SerialException as e:
    print(f"Error: Could not open serial port {SERIAL_PORT}. Please check the connection and permissions.")
    print(f"Details: {e}")
    exit()

# --- HRV Calculation Function ---
def calculate_hrv(all_peak_times):
    """
    Calculates Heart Rate Variability (HRV) metrics from a list of peak timestamps.

    Args:
        all_peak_times (list): A list of timestamps (in seconds) where peaks occurred.

    Returns:
        dict: A dictionary containing SDNN and RMSSD values in milliseconds.
              Returns None if there are not enough peaks to calculate HRV.
    """
    # Need at least 3 peaks to calculate 2 intervals for RMSSD
    if len(all_peak_times) < 3:
        return None

    # Calculate Inter-Beat Intervals (IBI) in milliseconds
    # This is the time difference between consecutive peaks
    ibi_list_ms = np.diff(all_peak_times) * 1000

    if len(ibi_list_ms) < 2:
        return None

    # 1. SDNN (Standard Deviation of NN intervals)
    # Measures the overall variability of heart rate.
    sdnn = np.std(ibi_list_ms)

    # 2. RMSSD (Root Mean Square of Successive Differences)
    # Measures the short-term, beat-to-beat variability.
    successive_diffs = np.diff(ibi_list_ms)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))

    return {"sdnn": sdnn, "rmssd": rmssd}


# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot([], [], 'b-', label='PPG Signal')
peaks_plot, = ax.plot([], [], 'rx', markersize=10, label='Detected Peaks')

# --- Animation Function ---
def animate(i):
    """
    This function is called periodically by FuncAnimation to update the plot.
    """
    global peak_times, latest_hrv

    try:
        # Read a line of data from the serial port
        serial_data = ser.readline().decode('utf-8').strip()

        # Proceed only if data was received and is in the correct format
        if serial_data and serial_data.startswith('>signal:'):
            # Expected format: >signal:106014
            parts = serial_data.split(':')
            if len(parts) == 2:
                # Clean and convert data
                signal_str = parts[1].strip()
                
                # Generate a timestamp since it's not provided by the device
                timestamp = time.time() 
                signal = int(signal_str)

                # Append new data to our deques
                timestamps.append(timestamp)
                ppg_signals.append(signal)

    except (ValueError, IndexError) as e:
        # Skip corrupted lines
        print(f"Warning: Could not parse serial data '{serial_data}'. Error: {e}")
        return line, peaks_plot,
    except KeyboardInterrupt:
        print("Stopping script...")
        ser.close()
        plt.close()
        exit()


    # Don't plot if we don't have data
    if not timestamps:
        return line, peaks_plot,

    # --- Peak Detection ---
    # Convert deque to numpy array for processing
    signal_array = np.array(ppg_signals)
    # Use SciPy to find peaks based on configured height and distance
    peak_indices, _ = find_peaks(signal_array, height=PEAK_MIN_HEIGHT, distance=PEAK_MIN_DISTANCE)

    # Get the actual signal values and timestamps for the detected peaks
    peak_s = [ppg_signals[i] for i in peak_indices]
    peak_t = [timestamps[i] for i in peak_indices]
    
    # Update the global list of peak times for HRV calculation
    # Add only new peaks that are not already in the list
    for t in peak_t:
        if t not in peak_times:
            peak_times.append(t)
            # Optional: Keep the peak_times list from growing indefinitely
            # if len(peak_times) > 100: peak_times.pop(0)

    # --- HRV Calculation Trigger ---
    # Calculate HRV every 10 new peaks to get a stable reading
    if len(peak_times) > 1 and len(peak_times) % 10 == 0:
        hrv_results = calculate_hrv(peak_times)
        if hrv_results:
            latest_hrv = hrv_results


    # --- Update Plot ---
    # Update data for the PPG signal line and the peak markers
    line.set_data(range(len(timestamps)), ppg_signals)
    peaks_plot.set_data(peak_indices, peak_s)
    
    # Dynamically adjust plot limits
    ax.set_xlim(0, DATA_WINDOW_SIZE - 1)
    if ppg_signals:
        min_val = min(ppg_signals)
        max_val = max(ppg_signals)
        y_margin = (max_val - min_val) * 0.1 # Add 10% margin
        ax.set_ylim(min_val - y_margin, max_val + y_margin)

    # Update labels and title
    ax.set_xlabel("Data Point Index (Moving Window)")
    ax.set_ylabel("PPG Signal Amplitude")
    ax.set_title(
        f"Real-time PPG Signal\n"
        f"Latest HRV - SDNN: {latest_hrv['sdnn']:.2f} ms | RMSSD: {latest_hrv['rmssd']:.2f} ms"
    )
    ax.legend(loc='upper left')
    ax.grid(True)

    return line, peaks_plot,

# --- Main Execution ---
if __name__ == '__main__':
    try:
        # Create the animation
        ani = animation.FuncAnimation(fig, animate, blit=True, interval=20, save_count=10)
        
        # Display the plot
        plt.show()

    finally:
        # Ensure the serial port is closed when the script ends or is interrupted
        if ser.is_open:
            ser.close()
            print("Serial port closed.")
