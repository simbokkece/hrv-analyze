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
MQTT_SUBSCRIBE_TOPIC = "mod_server/811/cmd"
username = "pod_0001"
password = "pod_0001"
MQTT_CLIENT_ID = f"hrv-client-{int(time.time())}"


# --- (Add this new function) ---

def get_port_from_mqtt(broker_host, broker_port, username, password):
    """
    Connects to MQTT and waits for a retained message
    on the config topic to get the serial port.
    """
    print("--- Connecting to MQTT to get serial port configuration ---")
    
    # We need a shared variable for the callback to set
    port_config = {"port": None}
    
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"Successfully connected to MQTT broker at {broker_host}...")
            # Subscribe to the config topic
            client.subscribe(MQTT_SUBSCRIBE_TOPIC.replace("command", "mod_server/811/port"))
        else:
            print(f"Failed to connect to MQTT: {rc}")

    def on_message(client, userdata, msg):
        """Called when the port message is received."""
        try:
            port = msg.payload.decode('utf-8').strip()
            if port.startswith('/dev/tty') or port.startswith('COM'):
                print(f"--- Received serial port: {port} ---")
                port_config["port"] = port
                # We got what we needed, stop the client's loop
                client.loop_stop()
            else:
                print(f"Received non-serial-port message: {port}")
        except Exception as e:
            print(f"Error processing config message: {e}")

    # Use a unique client ID for this one-time task
    config_client_id = f"hrv-config-fetcher-{int(time.time())}"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, config_client_id)
    client.username_pw_set(username, password)
    
    # Assign the callbacks
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(broker_host, broker_port, 60)
        # client.loop_forever() blocks, so we use loop_start()
        # and a manual wait loop.
        
        # But wait! If the message is retained, on_message
        # will be called *before* loop_start() even returns.
        # A safer pattern is loop_start() and a wait.
        
        client.loop_start()
        
        print("Waiting for serial port configuration from MQTT...")
        timeout_start = time.time()
        while port_config["port"] is None:
            time.sleep(0.1)
            # Timeout after 30 seconds
            if time.time() - timeout_start > 30:
                print("Error: Timed out waiting for port configuration.")
                client.loop_stop()
                return None
        
        # We got the port, loop was stopped by on_message
        return port_config["port"]

    except Exception as e:
        print(f"Error connecting to MQTT for config: {e}")
        if client.is_connected():
            client.loop_stop()
        return None

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

# def serial_write_thread(ser, running_event):
#     """
#     Waits for user input in a separate thread and sends commands to the serial port.
#     """
#     print("\n--- Serial Write Thread Started ---")
#     print("Type 's' and press Enter to send the command.")
    
#     while running_event.is_set():
#         try:
#             # This input() call will block the *write thread*
#             # without stopping the main loop.
#             command_key = input() 
            
#             if not running_event.is_set():
#                 break

#             if command_key.lower() == 's':
#                 # --- This is your command ---
#                 command_payload = '${"id":811,"cmd":81101,"val":1};'
#                 # --- --- --- --- --- --- ---

#                 # Convert to a JSON string
#                 # command_json = json.dumps(command_payload)
                
#                 # Add a newline, as most serial devices expect it
#                 # command_to_send = command_json + '\n' 
                
#                 # Encode to bytes and send
#                 ser.write(command_payload.encode('utf-8'))
                
#                 print(f"--> [SENT]: {command_payload}")
#             elif command_key.lower() == 'o':
#                 # --- This is your command ---
#                 command_payload = '${"id":811,"cmd":81101,"val":0};'
#                 # --- --- --- --- --- --- ---

#                 # Convert to a JSON string
#                 # command_json = json.dumps(command_payload)
                
#                 # Add a newline, as most serial devices expect it
#                 # command_to_send = command_json + '\n' 
                
#                 # Encode to bytes and send
#                 ser.write(command_payload.encode('utf-8'))
                
#                 print(f"--> [SENT]: {command_payload}")
#             else:
                
#                 print(f"(Type 's' and Enter to send command. You typed: '{command_key}')")
                
#         except EOFError:
#             # This can happen when the main program is closing
#             break
#         except Exception as e:
#             if running_event.is_set():
#                 print(f"Error in write thread: {e}")
#             break
#     print("--- Serial Write Thread Stopping ---")


# MODIFIED FUNCTION: Listens to MQTT for commands and writes to serial
def serial_write_thread(ser, mqtt_client, running_event):
    """
    Subscribes to an MQTT topic and sends specific commands to the serial port.
    This function runs in a separate thread.
    """
    print("\n--- Serial Write Thread Started ---")
    
    # --- Define the MQTT On-Message Callback ---
    # This function will be called by the Paho client's loop
    # when a message arrives on a subscribed topic.
    def on_message_callback(client, userdata, msg):
        """Processes incoming MQTT messages."""
        if not running_event.is_set():
            return # Main program is shutting down

        try:
            # 1. Get the raw message payload as a string
            line_data = msg.payload.decode('utf-8').strip()
            
            # 2. Try to parse it as JSON
            data = json.loads(line_data)

            # 3. Check if it's one of the specific commands you want
            if data == {"id": 811, "cmd": 81101, "val": 0} or \
               data == {"id": 811, "cmd": 81101, "val": 1}:
                
                # 4. Format the command (using the original string)
                # Encapsulate with $ and ; and add a newline
                command_to_send = f"${line_data};\n" 
                
                # 5. Send over serial
                if ser and ser.is_open:
                    ser.write(command_to_send.encode('utf-8'))
                    print(f"--> [MQTT->SERIAL]: Sent command: {line_data}")
                else:
                    print("--> [MQTT->SERIAL]: Serial port not open. Cannot send.")
            
            else:
                # It was valid JSON, but not a command we care about
                print(f"--> [MQTT RX]: (Skipped) {line_data}")
                
        except json.JSONDecodeError:
            # Not valid JSON
            print(f"--> [MQTT RX]: (Invalid JSON received) {msg.payload.decode('utf-8')}")
        except Exception as e:
            print(f"Error in on_message_callback: {e}")

    # --- End of Callback Definition ---

    if not mqtt_client:
        print("Write thread has no MQTT client. Stopping.")
        return

    try:
        # 1. Attach the callback function to the client
        # This tells Paho "call on_message_callback whenever a message comes in"
        mqtt_client.on_message = on_message_callback
        
        # 2. Subscribe to the command topic
        result, mid = mqtt_client.subscribe(MQTT_SUBSCRIBE_TOPIC)
        if result == mqtt.MQTT_ERR_SUCCESS:
            print(f"Successfully subscribed to MQTT topic '{MQTT_SUBSCRIBE_TOPIC}'")
        else:
            print(f"Failed to subscribe to '{MQTT_SUBSCRIBE_TOPIC}'. Error: {result}")

        # 3. Wait for the main loop to signal shutdown
        # The Paho loop (client.loop_start()) is already running in its own
        # background thread. This thread just needs to stay alive to
        # keep the 'on_message_callback' in scope and wait for the end.
        running_event.wait() # This blocks until running_event.clear() is called
    
    except Exception as e:
        if running_event.is_set():
            print(f"Error in write thread: {e}")
    
    print("--- Serial Write Thread Stopping ---")


# MODIFIED FUNCTION: Now accepts 'port' as an argument
def main(port, show_plot):
    """Main function to run the real-time HRV monitoring loop."""
    # The 'port' from the command line is passed here
    ser = initialize_serial(port)
    if not ser:
        return

    mqtt_client = setup_mqtt_client()
    if not mqtt_client:
        print("Continuing without MQTT (but command-listener thread will not work).")

    # --- Threading Setup ---
    running_event = threading.Event()
    running_event.set() # Set the event, similar to running = True

    # Start the write thread
    write_thread = threading.Thread(
        target=serial_write_thread, 
        args=(ser, mqtt_client, running_event)
    )
    write_thread.start()
    # --- End Threading Setup ---

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
    
    # running = True
    try:
        while running_event.is_set():
            if show_plot and not plt.fignum_exists(fig.number):
                running_event.clear() # <-- MODIFIED
                continue

            if ser.in_waiting > 0:
                try:
                    line_data = ser.readline().decode('utf-8').strip()
                    data = json.loads(line_data)

                    # if ':' in line_data:
                    #     signal_value = float(line_data.split(':')[1])
                    #     current_time = int(time.time() * 1000)

                    #     if start_time_ms is None:
                    #         start_time_ms = current_time

                    #     timestamps_ms.append(current_time)
                    #     ppg_signal.append(signal_value)
                
                    if "signal" in data:
                        # Ensure the value is a float (or can be converted)
                        signal_value = float(data["signal"])
                        # print(signal_value) 
                        
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

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n--- Monitoring stopped by user ---")
        running_event.clear() # <-- MODIFIED: Signal threads to stop
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        running_event.clear() # <-- MODIFIED: Signal threads to stop
    finally:
        if show_plot:
            plt.ioff()
        
        # --- Wait for write thread to finish ---
        print("Waiting for write thread to close...")
        write_thread.join(timeout=2.0) # Wait for the thread to exit
        print("Write thread closed.")
        # --- --- --- --- --- --- --- --- --- ---

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
    
    # # NEW ARGUMENT: For specifying the serial port
    # parser.add_argument("--port", type=str, required=True,
    #                     help="The serial port to connect to (e.g., /dev/ttyUSB0 or COM3).")
    
    parser.add_argument("--no-plot", action="store_false", dest="show_plot",
                        help="Run the script in headless mode without displaying the plot.")
    args = parser.parse_args()

    serial_port = get_port_from_mqtt(
        broker_host=MQTT_BROKER_HOST,
        broker_port=MQTT_BROKER_PORT,
        username=username,
        password=password
    )

    if serial_port:
        # MODIFIED CALL: Pass the port received from MQTT
        print(f"Starting main application with port {serial_port}")
        main(port=serial_port, show_plot=args.show_plot)
    else:
        print("Error: Could not determine serial port. Exiting.")

    # # MODIFIED CALL: Pass the 'port' argument to the main function
    # main(port=args.port, show_plot=args.show_plot)
