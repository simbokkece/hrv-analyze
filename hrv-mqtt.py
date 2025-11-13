import serial
import numpy as np
from scipy.signal import find_peaks
from collections import deque
import time
import json
import argparse # Used for command-line arguments
import threading
import os

# --- For MQTT ---
import paho.mqtt.client as mqtt

# --- Plotting Imports (Loaded conditionally later) ---
plt = None
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# --- End Plotting Imports ---

# --- Configuration ---
# User-configurable settings
SERIAL_PORT = '/dev/esp32_811'
BAUD_RATE = 230400
DATA_WINDOW_SECONDS = 10
MIN_HRV_INTERVALS = 10
MAX_HRV_INTERVALS = 30

# --- Signal Processing Parameters ---
SAMPLING_RATE_HZ = 12500

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "hrv/data"
MQTT_SUBSCRIBE_TOPIC = "mod_server/811/cmd"
MQTT_HEARTBEAT_TOPIC = "mod_server/811/data"
username = "pod_0001"
password = "pod_0001"
MQTT_CLIENT_ID = f"hrv-client-{int(time.time())}"


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

def serial_write_thread(ser, mqtt_client, running_event):
    """
    Subscribes to an MQTT topic and sends specific commands to the serial port.
    This function runs in a separate thread.
    """
    print("\n--- Serial Write Thread Started ---")
    
    def on_message_callback(client, userdata, msg):
        """Processes incoming MQTT messages."""
        if not running_event.is_set():
            return 

        try:
            line_data = msg.payload.decode('utf-8').strip()
            data = json.loads(line_data)

            if data == {"id": 811, "cmd": 81101, "val": 0} or \
               data == {"id": 811, "cmd": 81101, "val": 1}:
                
                command_to_send = f"${line_data};\n" 
                
                if ser and ser.is_open:
                    ser.write(command_to_send.encode('utf-8'))
                    print(f"--> [MQTT->SERIAL]: Sent command: {line_data}")
                else:
                    print("--> [MQTT->SERIAL]: Serial port not open. Cannot send.")
            
            else:
                print(f"--> [MQTT RX]: (Skipped) {line_data}")
                
        except json.JSONDecodeError:
            print(f"--> [MQTT RX]: (Invalid JSON received) {msg.payload.decode('utf-8')}")
        except Exception as e:
            print(f"Error in on_message_callback: {e}")

    if not mqtt_client:
        print("Write thread has no MQTT client. Stopping.")
        return

    try:
        mqtt_client.on_message = on_message_callback
        
        result, mid = mqtt_client.subscribe(MQTT_SUBSCRIBE_TOPIC)
        if result == mqtt.MQTT_ERR_SUCCESS:
            print(f"Successfully subscribed to MQTT topic '{MQTT_SUBSCRIBE_TOPIC}'")
        else:
            print(f"Failed to subscribe to '{MQTT_SUBSCRIBE_TOPIC}'. Error: {result}")

        running_event.wait() 
    
    except Exception as e:
        if running_event.is_set():
            print(f"Error in write thread: {e}")
    
    print("--- Serial Write Thread Stopping ---")


def main(port, show_plot):
    """Main function to run the real-time HRV monitoring loop."""
    
    # --- MODIFICATION: Conditionally import plotting libraries ---
    global plt
    fig, ax, line_raw, peaks_plot = (None, None, None, None)

    if show_plot:
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            print("Plotting enabled.")
        except ImportError:
            print("Warning: matplotlib or tkinter not found. Disabling plotting.")
            show_plot = False # Force disable
    else:
        print("Plotting disabled.")
    # --- END MODIFICATION ---

    ser = initialize_serial(port)
    if not ser:
        print("Serial port is not correct")
        return

    mqtt_client = setup_mqtt_client()
    if not mqtt_client:
        print("Continuing without MQTT (but command-listener thread will not work).")

    running_event = threading.Event()
    running_event.set() 

    write_thread = threading.Thread(
        target=serial_write_thread, 
        args=(ser, mqtt_client, running_event)
    )
    write_thread.start()

    max_len_data = int(200)
    timestamps_ms = deque(maxlen=max_len_data)
    ppg_signal = deque(maxlen=max_len_data)
    nn_intervals = deque(maxlen=MAX_HRV_INTERVALS)
    last_peak_time = None
    start_time_ms = None

    # --- Heartbeat variables ---
    heartbeat_count = 0
    last_heartbeat_time = time.time()
    # --- End Heartbeat ---

    # This setup is now conditional
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
    print("Waiting for data... Ensure your device is sending data in '{\"signal\": value}' format.")
    
    try:
        while running_event.is_set():
            if show_plot and (fig is None or not plt.fignum_exists(fig.number)):
                # Check if fig is None or plot window was closed
                print("Plot window closed by user or failed to create.")
                running_event.clear() 
                continue

            if ser.in_waiting > 0:
                try:
                    line_data = ser.readline().decode('utf-8').strip()
                    data = json.loads(line_data)
                
                    if "signal" in data:
                        signal_value = float(data["signal"])
                        current_time = int(time.time() * 1000)

                        if start_time_ms is None:
                            start_time_ms = current_time

                        timestamps_ms.append(current_time)
                        ppg_signal.append(signal_value)

                except (json.JSONDecodeError, ValueError, IndexError, UnicodeDecodeError, KeyError):
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

                latest_plot_time = plot_time_s[-1] if len(plot_time_s) > 0 else 0
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
                                "id": 811,
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

            # --- Heartbeat Logic ---
            current_loop_time = time.time()
            if (current_loop_time - last_heartbeat_time) >= 1.0:
                heartbeat_count += 1
                payload = {
                    "id": 811,
                    "ts": int(current_loop_time),
                    "uptime_sec": heartbeat_count
                }
                json_payload = json.dumps(payload)
                
                if mqtt_client:
                    result = mqtt_client.publish(MQTT_HEARTBEAT_TOPIC, json_payload)
                    if result.rc != mqtt.MQTT_ERR_SUCCESS:
                        print(f"Failed to publish heartbeat. Error: {result.rc}")
                
                last_heartbeat_time = current_loop_time
            # --- End Heartbeat Logic ---

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n--- Monitoring stopped by user ---")
        running_event.clear() 
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        running_event.clear() 
    finally:
        if show_plot and plt:
            plt.ioff()
        
        print("Waiting for write thread to close...")
        write_thread.join(timeout=2.0) 
        print("Write thread closed.")

        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            print("MQTT client disconnected.")
        print("Script finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time HRV monitoring from a serial device with MQTT publishing.")
    
    # This logic is correct:
    # Default is show_plot=True.
    # --no-plot sets show_plot=False.
    parser.add_argument("--no-plot", action="store_false", dest="show_plot",
                        help="Run the script in headless mode without displaying the plot.")
    args = parser.parse_args()

    print(f"Starting main application with port {SERIAL_PORT}")
    main(port=SERIAL_PORT, show_plot=args.show_plot)