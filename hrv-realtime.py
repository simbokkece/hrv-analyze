import serial
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- Configuration ---
# User-configurable settings
SERIAL_PORT = '/dev/ttyUSB0'  # Windows: 'COM3', 'COM4', etc. Linux: '/dev/ttyUSB0', '/dev/ttyACM0', etc.
BAUD_RATE = 460800
DATA_WINDOW_SECONDS = 10  # Analyze the last 10 seconds of data
MIN_HRV_INTERVALS = 10    # Minimum number of intervals needed to calculate HRV
MAX_HRV_INTERVALS = 30    # Keep the last 30 intervals for a rolling HRV calculation

# --- Signal Processing Parameters ---
# These parameters are for the bandpass filter and peak detection
SAMPLING_RATE_HZ = 12500  # IMPORTANT: This should match the rate your sensor sends data (e.g., 100 samples per second)
LOW_CUT_HZ = 0.7        # Low cutoff for the bandpass filter (in Hz)
HIGH_CUT_HZ = 8.0       # High cutoff for the bandpass filter (in Hz)

def initialize_serial():
    """Tries to connect to the serial port and returns the serial object."""
    try:
        # The 'pyserial' library provides the serial.Serial class
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        time.sleep(2) # Wait for the serial connection to initialize
        return ser
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}.")
        print(f"Details: {e}")
        print("Please check the port name, make sure the device is connected, and that no other program is using it.")
        return None

def main():
    """Main function to run the real-time HRV monitoring loop."""
    ser = initialize_serial()
    if not ser:
        return

    # --- Data Storage Setup ---
    max_len_data = int(200)
    timestamps_ms = deque(maxlen=max_len_data)
    ppg_signal = deque(maxlen=max_len_data)
    nn_intervals = deque(maxlen=MAX_HRV_INTERVALS)
    last_peak_time = None

    start_time_ms = None

    # --- Plotting Setup ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # MODIFICATION: Create lines for both raw and filtered signals
    line_raw, = ax.plot([], [], 'orange', alpha=0.5, label='Raw Signal')
    # line_filtered, = ax.plot([], [], 'b-', label='Filtered Signal')
    peaks_plot, = ax.plot([], [], 'rx', markersize=10, label='Detected Peaks')
    
    ax.set_title("Real-Time PPG Signal (30-Second Window)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal Amplitude")
    ax.grid(True)
    ax.legend() # Add legend to identify the lines

    print("\n--- Starting Real-Time HRV Monitoring ---")
    print("Waiting for data... Ensure your device is sending data in 'signal:value' format.")
    print("Close the plot window to stop the script.")

    try:
        while plt.fignum_exists(fig.number):
            # 1. Read and Parse Data
            if ser.in_waiting > 0:
                try:
                    line_data = ser.readline().decode('utf-8').strip()
                    if ':' in line_data:
                        signal_value = float(line_data.split(':')[1])
                        current_time = int(time.time() * 1000)

                        # Record the timestamp of the very first data point received.
                        if start_time_ms is None:
                            start_time_ms = current_time

                        timestamps_ms.append(current_time)
                        ppg_signal.append(signal_value)
                    else:
                        continue
                except (ValueError, IndexError):
                    continue
            
            if len(ppg_signal) < 50:
                plt.pause(0.01)
                continue

            # 2. Process Signal
            current_signal_window = np.array(ppg_signal)
            # nyquist = 0.5 * SAMPLING_RATE_HZ
            # low = LOW_CUT_HZ / nyquist
            # high = HIGH_CUT_HZ / nyquist
            # b, a = butter(2, [low, high], btype='band')
            # filtered_signal = filtfilt(b, a, current_signal_window)

            # 3. Find Peaks
            min_peak_distance = 8
            min_prominence = 20
            peaks, _ = find_peaks(current_signal_window, 
                                  prominence=min_prominence,
                                  width=(1, 7), 
                                  distance=min_peak_distance)

            # 4. Update Plot
            # relative_time_s = (np.array(timestamps_ms) - timestamps_ms[0]) / 1000.0
            plot_time_s = (np.array(timestamps_ms) - start_time_ms) / 1000.0
            
            # MODIFICATION: Set data for both raw and filtered lines
            line_raw.set_data(plot_time_s, current_signal_window)
            # line_filtered.set_data(plot_time_s, filtered_signal)
            
            if len(peaks) > 0:
                peaks_plot.set_data(plot_time_s[peaks], current_signal_window[peaks])
            else:
                peaks_plot.set_data([], [])

            # print("Current Signal Window:", current_signal_window)
            
            # Set moving x-axis and dynamic y-axis
            latest_plot_time = plot_time_s[-1]
            ax.set_xlim(latest_plot_time - DATA_WINDOW_SECONDS, latest_plot_time)
            
            if len(current_signal_window) > 0:
                min_y = np.min(current_signal_window)
                max_y = np.max(current_signal_window)
                data_range = max_y - min_y
                
                # Add 10% padding to the top and bottom for visual clarity.
                # Use a default padding if the range is zero (flat line).
                padding = data_range * 0.10 if data_range > 0 else 1.0
                
                ax.set_ylim(min_y - padding, max_y + padding)
            
            fig.canvas.draw()
            fig.canvas.flush_events()

            # 5. Calculate HRV
            if len(peaks) > 0:
                latest_peak_time = timestamps_ms[peaks[-1]]
                if last_peak_time is None or latest_peak_time > last_peak_time:
                    if last_peak_time is not None:
                        interval = latest_peak_time - last_peak_time
                        if 300 < interval < 2000:
                            nn_intervals.append(interval)
                            print(f"Beat Detected! Interval: {interval} ms")
                        else:
                            print(f"Beat Rejected! (Interval {interval} ms out of range)")
                    last_peak_time = latest_peak_time

                    if len(nn_intervals) >= MIN_HRV_INTERVALS:
                        sdnn = np.std(nn_intervals)
                        successive_diffs = np.diff(nn_intervals)
                        rmssd = np.sqrt(np.mean(successive_diffs**2))
                        print("----------------------------------------")
                        print(f"HRV (last {len(nn_intervals)} beats):")
                        print(f"  SDNN:  {sdnn:.2f} ms")
                        print(f"  RMSSD: {rmssd:.2f} ms")
                        print("----------------------------------------")
            
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n--- Monitoring stopped by user ---")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        plt.ioff()
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")
        print("Script finished.")

if __name__ == '__main__':
    main()