import serial
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from scipy.signal import find_peaks

# --- Configuration ---
SERIAL_PORT = '/dev/ttyUSB0' # Change to your port
BAUD_RATE = 460800
PLOT_WINDOW_SECONDS = 10   # How many seconds of data to display
ANALYSIS_WINDOW_SECONDS = 5 # How many seconds of data to analyze for peaks

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Serial port opened successfully.")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return
        
    time.sleep(2)

    # Use a longer maxlen to ensure we have enough data for the analysis window
    buffer_size = int(PLOT_WINDOW_SECONDS * 100) # Assuming a max sample rate of 100Hz
    time_buffer = deque(maxlen=buffer_size)
    data_buffer = deque(maxlen=buffer_size)

    # For storing the final, correct peaks
    peak_time_buffer = deque(maxlen=50) 
    peak_val_buffer = deque(maxlen=50)

    # Setup plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], '-o', markersize=3, label='ADC Data')
    peaks_plot, = ax.plot([], [], 'rx', markersize=10, label='Detected Peaks')
    ax.legend()
    ax.grid(True)
    ax.set_title("Real-Time PPG with Prominence-Based Peak Detection")

    last_analysis_time = 0

    print("Starting real-time analysis...")
    while True:
        try:
            if not plt.fignum_exists(fig.number):
                print("Plot window closed. Exiting.")
                break

            line_data = ser.readline().decode('utf-8').strip()
            if ':' in line_data:
                current_val = float(line_data.split(':')[1])
                current_time = time.time()
                data_buffer.append(current_val)
                time_buffer.append(current_time)

                # --- NEW ANALYSIS LOGIC ---
                # Analyze every half second instead of every single data point
                if current_time - last_analysis_time > 0.5 and len(data_buffer) > 50:
                    last_analysis_time = current_time
                    
                    # Convert to numpy arrays for analysis
                    times_np = np.array(time_buffer)
                    data_np = np.array(data_buffer)
                    
                    # --- NORMALIZE the signal to make prominence work reliably ---
                    normalized_data = (data_np - np.mean(data_np)) / np.std(data_np)
                    
                    # --- FIND PEAKS USING PROMINENCE ---
                    # We need the sampling rate for the distance parameter
                    fs = 1.0 / np.mean(np.diff(times_np))
                    min_dist_samples = int(fs * 0.4) # Min distance = 400ms (max 150 BPM)

                    # This is the key change: use prominence to find only the main peaks
                    peaks, _ = find_peaks(
                        normalized_data, 
                        prominence=0.6, # KEY PARAMETER: Peak must be > 0.8 standard deviations prominent
                        distance=min_dist_samples
                    )
                    
                    # Store the found peaks for plotting
                    if len(peaks) > 0:
                        peak_time_buffer.clear()
                        peak_val_buffer.clear()
                        peak_time_buffer.extend(times_np[peaks])
                        peak_val_buffer.extend(data_np[peaks])

                # --- Plotting Logic (remains mostly the same) ---
                if len(time_buffer) > 1:
                    line.set_data(time_buffer, data_buffer)
                    peaks_plot.set_data(peak_time_buffer, peak_val_buffer)

                    latest_time = time_buffer[-1]
                    ax.set_xlim(latest_time - PLOT_WINDOW_SECONDS, latest_time)

                    visible_indices = np.array(time_buffer) > (latest_time - PLOT_WINDOW_SECONDS)
                    if np.any(visible_indices):
                        visible_data = np.array(data_buffer)[visible_indices]
                        min_y, max_y = np.min(visible_data), np.max(visible_data)
                        padding = (max_y - min_y) * 0.1 or 100
                        ax.set_ylim(min_y - padding, max_y + padding)

                    fig.canvas.draw()
                    fig.canvas.flush_events()

        except (UnicodeDecodeError, ValueError):
            continue
        except KeyboardInterrupt:
            print("Stopping...")
            break

    ser.close()
    print("Serial port closed.")

if __name__ == "__main__":
    main()