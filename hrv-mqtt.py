import serial
import numpy as np
from scipy.signal import find_peaks
from collections import deque
import time
import json
import argparse # Used for command-line arguments

# --- For MQTT ---
import paho.mqtt.client as mqtt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- Configuration ---
# User-configurable settings
# SERIAL_PORT = '/dev/ttyUSB7' # <-- THIS LINE IS REMOVED
BAUD_RATE = 460800
DATA_WINDOW_SECONDS = 10
MIN_HRV_INTERVALS = 10
MAX_HRV_INTERVALS = 30

# --- Signal Processing Parameters ---
SAMPLING_RATE_HZ = 12500

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "hrv/data"
username = "pod_0001"
password = "pod_0001"
MQTT_CLIENT_ID = f"hrv-client-{int(time.time())}"

# MODIFIED FUNCTION: Now accepts 'port' as an argument
def initialize_serial(port):
    """Tries to connect to the specified serial port and returns the serial object."""
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"Successfully connected to {port} at {BAUD_RATE} baud.")
        time.sleep(2)
        return ser
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {port}.")
        print(f"Details: {e}")
        return None

# --- MQTT Setup Function ---
def setup_mqtt_client():
    """Creates, configures, and connects the MQTT client."""
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, MQTT_CLIENT_ID)
        client.username_pw_set(username, password)
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        client.loop_start()
        print(f"Successfully connected to MQTT Broker at {MQTT_BROKER_HOST}")
        return client
    except Exception as e:
        print(f"Error: Could not connect to MQTT Broker.")
        print(f"Details: {e}")
        return None

# MODIFIED FUNCTION: Now accepts 'port' as an argument
def main(port, show_plot):
    """Main function to run the real-time HRV monitoring loop."""
    # The 'port' from the command line is passed here
    ser = initialize_serial(port)
    if not ser:
        return

    mqtt_client = setup_mqtt_client()
    if not mqtt_client:
        print("Continuing without MQTT.")

    max_len_data = int(200)
    timestamps_ms = deque(maxlen=max_len_data)
    ppg_signal = deque(maxlen=max_len_data)
    nn_intervals = deque(maxlen=MAX_HRV_INTERVALS)
    last_peak_time = None
    start_time_ms = None

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
    
    running = True
    try:
        while running:
            if show_plot and not plt.fignum_exists(fig.number):
                running = False
                continue

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
                time.sleep(0.01)
                continue

            current_signal_window = np.array(ppg_signal)
            peaks, _ = find_peaks(current_signal_window, prominence=20, width=(1, 7), distance=8)

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
                        avg_interval_ms = np.mean(nn_intervals)
                        heart_rate_bpm = 60000.0 / avg_interval_ms

                        print("----------------------------------------")
                        print(f"HRV (last {len(nn_intervals)} beats):")
                        print(f"  Heart Rate: {heart_rate_bpm:.2f} BPM")
                        print(f"  SDNN:  {sdnn:.2f} ms")
                        print(f"  RMSSD: {rmssd:.2f} ms")
                        print("----------------------------------------")

                        if mqtt_client:
                            payload = {
                                "id": 810,
                                "ts": int(time.time()),
                                "bpm": round(heart_rate_bpm, 2),
                                "hrv": round(rmssd, 2),
                                "sdnn": round(sdnn, 2),
                                "rmssd": round(rmssd, 2),
                                "intervals_used": len(nn_intervals)
                            }
                            json_payload = json.dumps(payload)
                            result = mqtt_client.publish(MQTT_TOPIC, json_payload)
                            
                            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                                print(f"Successfully published to MQTT topic '{MQTT_TOPIC}'")
                            else:
                                print(f"Failed to publish to MQTT. Error code: {result.rc}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n--- Monitoring stopped by user ---")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
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
    
    # NEW ARGUMENT: For specifying the serial port
    parser.add_argument("--port", type=str, required=True,
                        help="The serial port to connect to (e.g., /dev/ttyUSB0 or COM3).")
    
    parser.add_argument("--no-plot", action="store_false", dest="show_plot",
                        help="Run the script in headless mode without displaying the plot.")
    args = parser.parse_args()

    # MODIFIED CALL: Pass the 'port' argument to the main function
    main(port=args.port, show_plot=args.show_plot)
