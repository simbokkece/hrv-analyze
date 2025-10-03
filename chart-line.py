import json
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def parse_log_file(file_path):
    """
    Parses a log file to extract timestamps and structured data.

    Args:
        file_path (str): The path to the log file.

    Returns:
        tuple: A tuple containing lists of timestamps and parsed data
               for heartbeat, evaporator, and condenser.
    """
    # Timestamps for each type of measurement
    hb_timestamps = []
    temp_timestamps = []

    # Data lists
    heartbeat_data = []
    evap_in_data = []
    evap_out_data = []
    cond_in_data = []
    cond_out_data = []

    # Regex to find the JSON payload in each log line
    json_pattern = re.compile(r'\{.*\}')

    print(f"Attempting to open and read '{file_path}'...")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Find the JSON part of the line
                match = json_pattern.search(line)
                if not match:
                    continue

                try:
                    # Load the JSON data
                    data = json.loads(match.group(0))
                    
                    # Extract timestamp and convert to datetime object
                    timestamp_str = line.split('Z')[0]
                    dt_object = datetime.strptime(timestamp_str, '%Y/%m/%d %H:%M:%S')

                    # Check for heartbeat data
                    if 'hb' in data:
                        hb_timestamps.append(dt_object)
                        heartbeat_data.append(data['hb'])
                    # Check for temperature data
                    elif 'comp' in data:
                        temp_timestamps.append(dt_object)
                        # Append evaporator data
                        if 'evap' in data:
                            evap_in_data.append(data['evap'].get('in'))
                            evap_out_data.append(data['evap'].get('out'))
                        # Append condenser data
                        if 'cond' in data:
                            cond_in_data.append(data['cond'].get('in'))
                            cond_out_data.append(data['cond'].get('out'))

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Skipping malformed line: {line.strip()} | Error: {e}")
        
        print("Successfully parsed the log file.")
        return (hb_timestamps, heartbeat_data, temp_timestamps, 
                evap_in_data, evap_out_data, cond_in_data, cond_out_data)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None, None, None, None, None


def plot_data(hb_ts, hb_data, temp_ts, evap_in, evap_out, cond_in, cond_out):
    """
    Plots the parsed log data into three separate line charts.

    Args:
        hb_ts (list): Timestamps for heartbeat data.
        hb_data (list): Heartbeat data points.
        temp_ts (list): Timestamps for temperature data.
        evap_in (list): Evaporator 'in' temperature data.
        evap_out (list): Evaporator 'out' temperature data.
        cond_in (list): Condenser 'in' temperature data.
        cond_out (list): Condenser 'out' temperature data.
    """
    if not hb_ts and not temp_ts:
        print("No data available to plot.")
        return

    # Create a figure with 3 subplots stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    fig.suptitle('Log Data Analysis', fontsize=20)

    # --- Plot 1: Heartbeat ---
    ax1.plot(hb_ts, hb_data, label='Heartbeat (hb)', color='blue', marker='.', linestyle='-')
    ax1.set_title('Heartbeat Over Time')
    ax1.set_ylabel('Heartbeat Value')
    ax1.legend()
    ax1.grid(True)

    # --- Plot 2: Evaporator Temperatures ---
    ax2.plot(temp_ts, evap_in, label='Evaporator In', color='green', marker='.', linestyle='-')
    ax2.plot(temp_ts, evap_out, label='Evaporator Out', color='darkgreen', marker='.', linestyle='--')
    ax2.set_title('Evaporator Temperatures Over Time')
    ax2.set_ylabel('Temperature (°C)')
    ax2.legend()
    ax2.grid(True)

    # --- Plot 3: Condenser Temperatures ---
    ax3.plot(temp_ts, cond_in, label='Condenser In', color='red', marker='.', linestyle='-')
    ax3.plot(temp_ts, cond_out, label='Condenser Out', color='darkred', marker='.', linestyle='--')
    ax3.set_title('Condenser Temperatures Over Time')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_xlabel('Time')
    ax3.legend()
    ax3.grid(True)
    
    # Format the x-axis to show time properly
    xfmt = mdates.DateFormatter('%H:%M:%S')
    ax3.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=45)

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
    plt.show()


if __name__ == "__main__":
    # Specify the path to your log file
    LOG_FILE_PATH = 'id506.log'
    
    # Parse the data from the log file
    (hb_ts, hb_data, temp_ts, evap_in, 
     evap_out, cond_in, cond_out) = parse_log_file(LOG_FILE_PATH)
    
    # If data was parsed successfully, plot it
    if hb_ts or temp_ts:
        plot_data(hb_ts, hb_data, temp_ts, evap_in, evap_out, cond_in, cond_out)
