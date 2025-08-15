import serial
import numpy as np
from scipy.signal import find_peaks
from collections import deque
import time
import json
import argparse # Added for command-line arguments

# --- Added for MQTT ---
import paho.mqtt.client as mqtt

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
SAMPLING_RATE_HZ = 12500
LOW_CUT_HZ = 0.7
HIGH_CUT_HZ = 8.0

# --- MQTT Configuration (NEW SECTION) ---
MQTT_BROKER_HOST = "localhost" # Use 'localhost' if you have a local broker
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "hrv/data"
# A unique client ID is good practice, especially if multiple devices are running
MQTT_CLIENT_ID = f"hrv-client-{int(time.time())}"

def initialize_serial():
    """Tries to connect to the serial port and returns the serial object."""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        time.sleep(2)
        return ser
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}.")
        print(f"Details: {e}")
        return None

# --- MQTT Setup Function (NEW) ---
def setup_mqtt_client():
    """Creates, configures, and connects the MQTT client."""
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, MQTT_CLIENT_ID)
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        client.loop_start() # Starts a background thread to handle MQTT network traffic
        print(f"Successfully connected to MQTT Broker at {MQTT_BROKER_HOST}")
        return client
    except Exception as e:
        print(f"Error: Could not connect to MQTT Broker.")
        print(f"Details: {e}")
        return None

def main(show_plot):
    """Main function to run the real-time HRV monitoring loop."""
    ser = initialize_serial()
    if not ser:
        return

    # --- Initialize MQTT Client (NEW) ---
    mqtt_client = setup_mqtt_client()
    if not mqtt_client:
        print("Continuing without MQTT.")

    # --- Data Storage Setup ---
    max_len_data = int(200)
    timestamps_ms = deque(maxlen=max_len_data)
    ppg_signal = deque(maxlen=max_len_data)
    nn_intervals = deque(maxlen=MAX_HRV_INTERVALS)
    last_peak_time = None
    start_time_ms = None

    # --- Plotting Setup (Conditional) ---
    fig, ax, line_raw, peaks_plot = (None, None, None, None)
    if show_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 6))
        line_raw, = ax.plot([], [], 'orange', alpha=0.5, label='Raw Signal')
        peaks_plot, = ax.plot([], [], 'rx', markersize=10, label='Detected Peaks')
        ax.set_title("Real-Time PPG Signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal Amplitude")
        ax.grid(True)
        ax.legend()

    print("\n--- Starting Real-Time HRV Monitoring ---")
    print(f"Plotting enabled: {show_plot}")
    print("Waiting for data... Ensure your device is sending data in 'signal:value' format.")
    if show_plot:
        print("Close the plot window to stop the script.")
    else:
        print("Press CTRL+C to stop the script.")
    
    # --- Main Loop Condition (MODIFIED) ---
    # Loop continues if plotting is enabled and window is open, OR if plotting is disabled.
    running = True
    try:
        while running:
            # For plotting mode, check if the figure still exists
            if show_plot and not plt.fignum_exists(fig.number):
                running = False
                continue

            # 1. Read and Parse Data
            if ser.in_waiting > 0:
                try:
                    line_data = ser.readline().decode('utf-8').strip()
                    if ':' in line_data:
                        signal_value = float(line_data.split(':')[1])
                        current_time = int(time.time() * 1000)

                        if start_time_ms is None:
                            start_time_ms = current_time

                        timestamps_ms.append(current_time)
                        ppg_signal.append(signal_value)
                except (ValueError, IndexError, UnicodeDecodeError):
                    continue
            
            if len(ppg_signal) < 50:
                time.sleep(0.01) # Use time.sleep if not plotting
                continue

            # 2. Process Signal & 3. Find Peaks
            current_signal_window = np.array(ppg_signal)
            peaks, _ = find_peaks(current_signal_window, prominence=20, width=(1, 7), distance=8)

            # 4. Update Plot (Conditional)
            if show_plot:
                plot_time_s = (np.array(timestamps_ms) - start_time_ms) / 1000.0
                line_raw.set_data(plot_time_s, current_signal_window)
                if len(peaks) > 0:
                    peaks_plot.set_data(plot_time_s[peaks], current_signal_window[peaks])
                else:
                    peaks_plot.set_data([], [])

                latest_plot_time = plot_time_s[-1]
                ax.set_xlim(latest_plot_time - DATA_WINDOW_SECONDS, latest_plot_time)
                
                if len(current_signal_window) > 0:
                    min_y, max_y = np.min(current_signal_window), np.max(current_signal_window)
                    padding = (max_y - min_y) * 0.10 or 1.0
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
                    last_peak_time = latest_peak_time

                    if len(nn_intervals) >= MIN_HRV_INTERVALS:
                        sdnn = np.std(nn_intervals)
                        rmssd = np.sqrt(np.mean(np.diff(nn_intervals)**2))
                        
                        print("----------------------------------------")
                        print(f"HRV (last {len(nn_intervals)} beats):")
                        print(f"  SDNN:  {sdnn:.2f} ms")
                        print(f"  RMSSD: {rmssd:.2f} ms")
                        print("----------------------------------------")

                        # --- MQTT Publish (NEW) ---
                        if mqtt_client:
                            payload = {
                                "sdnn": round(sdnn, 2),
                                "rmssd": round(rmssd, 2),
                                "intervals_used": len(nn_intervals),
                                "timestamp_utc": int(time.time())
                            }
                            # Convert dictionary to JSON string
                            json_payload = json.dumps(payload)
                            # Publish to the topic
                            result = mqtt_client.publish(MQTT_TOPIC, json_payload)
                            
                            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                                print(f"Successfully published to MQTT topic '{MQTT_TOPIC}'")
                            else:
                                print(f"Failed to publish to MQTT. Error code: {result.rc}")

            # Pause briefly to prevent high CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n--- Monitoring stopped by user ---")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # --- Cleanup (MODIFIED) ---
        if show_plot:
            plt.ioff()
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            print("MQTT client disconnected.")
        print("Script finished.")

# --- Main execution block (MODIFIED for argument parsing) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time HRV monitoring from a serial device with MQTT publishing.")
    parser.add_argument("--no-plot", action="store_false", dest="show_plot",
                        help="Run the script in headless mode without displaying the plot.")
    args = parser.parse_args()

    main(show_plot=args.show_plot)