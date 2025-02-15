import subprocess
import os
import sys

# Get the directory of the current script (h2.py)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the energibridge executable relative to the script's directory
energibridge_path = os.path.join(script_dir, 'energibridge_things', 'energibridge')

# Define the command as a list
command = [
    energibridge_path,
    '-o', 'results.csv',
    '--summary',
    'timeout', '2'
]
print('updated version...')
# Run the command using subprocess.run
try:
    # Printing the command for debugging purposes
    print(f"Executing command: {' '.join(command)}")

    # Run the command and capture the output and errors
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    # Printing the output and errors to the console
    print("Command executed successfully!")
    print("Output:")
    print(result.stdout)  # Output from the command
    print("Error (if any):")
    print(result.stderr)  # Errors from the command, if any

except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e}")
    print(f"Exit code: {e.returncode}")
    print(f"Command output: {e.output}")
    print(f"Command stderr: {e.stderr}")
