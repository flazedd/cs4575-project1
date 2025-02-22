import psutil

# Retrieve temperature information (works on Linux/macOS)
temp_info = psutil.sensors_temperatures()

# Print CPU temperature (if available)
if "coretemp" in temp_info:
    print(f"CPU Temperature: {temp_info['coretemp'][0].current}°C")
else:
    print("Unable to retrieve CPU temperature.")
