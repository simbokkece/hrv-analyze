#!/usr/bin/env python3
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
import pandas as pd
from datetime import datetime

class PPGProcessor:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, window_size=1000):
        """
        Initialize the PPG processor.
        
        Args:
            port: Serial port to connect to
            baudrate: Baud rate for serial connection
            window_size: Number of samples to display in the plot window
        """
        self.port = port
        self.baudrate = baudrate
        self.window_size = window_size
        
        # Data storage
        self.timestamps = []
        self.ppg_data = []
        self.peak_indices = []
        self.peak_times = []
        self.rr_intervals = []  # in milliseconds
        
        # Plot setup
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.line1, = self.ax1.plot([], [], 'b-', label='PPG Signal')
        self.peaks_plot, = self.ax1.plot([], [], 'ro', label='Detected Peaks')
        self.rr_plot, = self.ax2.plot([], [], 'g-', label='RR Intervals')
        
        self.ax1.set_title('Real-time PPG Signal')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.legend(loc='upper right')
        
        self.ax2.set_title('RR Intervals')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('RR Interval (ms)')
        self.ax2.legend(loc='upper right')
        
        self.fig.tight_layout()
        
        # HRV metrics
        self.sdnn = 0
        self.rmssd = 0
        
        # Serial connection
        self.ser = None
        
    def connect_serial(self):
        """Establish serial connection"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate)
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"Failed to connect to serial port: {e}")
            return False
            
    def read_serial_data(self):
        """Read a line of data from serial port"""
        if self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                return float(line)
            except (ValueError, UnicodeDecodeError):
                return None
        return None
        
    def detect_peaks(self, data, timestamps):
        """
        Detect peaks in the PPG signal
        
        Args:
            data: PPG signal data
            timestamps: Corresponding timestamps
            
        Returns:
            peak_indices: Indices of detected peaks
        """
        # Parameters may need tuning based on your specific PPG signal
        height = np.mean(data) + 0.5 * np.std(data)  # Adaptive threshold
        distance = 40  # Minimum samples between peaks (adjust based on sampling rate)
        
        peak_indices, _ = find_peaks(data, height=height, distance=distance)
        return peak_indices
        
    def calculate_hrv(self):
        """Calculate HRV metrics from RR intervals"""
        if len(self.rr_intervals) > 1:
            # SDNN: Standard deviation of NN intervals
            self.sdnn = np.std(self.rr_intervals)
            
            # RMSSD: Root mean square of successive differences
            rr_diff = np.diff(self.rr_intervals)
            self.rmssd = np.sqrt(np.mean(rr_diff ** 2))
            
            print(f"HRV Metrics - SDNN: {self.sdnn:.2f} ms, RMSSD: {self.rmssd:.2f} ms")
        else:
            print("Not enough RR intervals to calculate HRV metrics")
            
    def update_plot(self, frame):
        """Update function for animation"""
        # Read data from serial
        value = self.read_serial_data()
        
        if value is not None:
            current_time = time.time()
            
            # Store data
            self.timestamps.append(current_time)
            self.ppg_data.append(value)
            
            # Keep only the window_size most recent samples
            if len(self.ppg_data) > self.window_size:
                self.ppg_data = self.ppg_data[-self.window_size:]
                self.timestamps = self.timestamps[-self.window_size:]
            
            # Normalize timestamps for display
            t_normalized = [t - self.timestamps[0] for t in self.timestamps]
            
            # Update PPG signal plot
            self.line1.set_data(t_normalized, self.ppg_data)
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            # Detect peaks
            if len(self.ppg_data) > 50:  # Wait for enough data
                peak_indices = self.detect_peaks(self.ppg_data, self.timestamps)
                
                # Update peak markers
                if len(peak_indices) > 0:
                    self.peaks_plot.set_data([t_normalized[i] for i in peak_indices], 
                                            [self.ppg_data[i] for i in peak_indices])
                    
                    # Process new peaks
                    new_peaks = [i for i in peak_indices if i not in self.peak_indices]
                    for idx in new_peaks:
                        self.peak_times.append(self.timestamps[idx])
                        
                        # Calculate RR intervals
                        if len(self.peak_times) > 1:
                            # Convert to milliseconds
                            rr = (self.peak_times[-1] - self.peak_times[-2]) * 1000
                            self.rr_intervals.append(rr)
                    
                    self.peak_indices = peak_indices
                    
                    # Update RR interval plot
                    if len(self.peak_times) > 1:
                        rr_times = [t - self.timestamps[0] for t in self.peak_times[1:]]
                        self.rr_plot.set_data(rr_times, self.rr_intervals)
                        self.ax2.relim()
                        self.ax2.autoscale_view()
                    
                    # Calculate HRV metrics every 10 seconds
                    if len(self.rr_intervals) > 5 and frame % 100 == 0:
                        self.calculate_hrv()
            
        return self.line1, self.peaks_plot, self.rr_plot
        
    def save_data(self, filename=None):
        """Save the collected data to a CSV file"""
        if filename is None:
            filename = f"ppg_hrv_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        if len(self.peak_times) > 1:
            # Create DataFrame with peak times and RR intervals
            df = pd.DataFrame({
                'peak_time': self.peak_times[1:],
                'rr_interval': self.rr_intervals,
            })
            
            # Add HRV metrics
            df['sdnn'] = self.sdnn
            df['rmssd'] = self.rmssd
            
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("Not enough data to save")
            
    def run(self):
        """Run the PPG processor"""
        if not self.connect_serial():
            print("Failed to connect to serial port. Exiting.")
            return
            
        # Set up animation
        ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True)
        
        # Display plot
        plt.show()
        
        # After plot is closed, save the data
        self.save_data()
        
        # Close serial connection
        if self.ser and self.ser.is_open:
            self.ser.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time PPG signal processing and HRV analysis')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate')
    parser.add_argument('--window', type=int, default=1000, help='Display window size (samples)')
    
    args = parser.parse_args()
    
    processor = PPGProcessor(port=args.port, baudrate=args.baudrate, window_size=args.window)
    processor.run()