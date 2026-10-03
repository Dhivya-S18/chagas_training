import serial
import csv
import time

PORT = "COM6"
BAUD = 115200
DURATION = 30

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Starting ECG recording...")
print(f"Recording for {DURATION} seconds...")

start = time.time()

with open("raw_ecg.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp_ms", "ecg_value"])

    while time.time() - start < DURATION:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if "," in line:
            try:
                timestamp, value = line.split(",")
                writer.writerow([int(timestamp), int(value)])
            except ValueError:
                pass

ser.close()

print("Recording complete.")
print("Saved as raw_ecg.csv")