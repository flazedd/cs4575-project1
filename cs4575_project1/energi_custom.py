import subprocess
import time
import os
import keyboard
import re

class EnergiCustom:
    def __init__(self, output="results.csv"):
        self.joules = None
        self.seconds = None
        self.process = None
        self.output = output

    def start(self):
        print(f'Output will be saved in {self.output} once you call stop()')
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the energibridge executable relative to the script's directory
        energibridge_path = os.path.join(script_dir, 'energibridge_things', 'energibridge')

        # Define the command as a list
        command = [
            energibridge_path,
            '-o', self.output,
            '--summary',
            'timeout', '99999'  # Maximum for windows, equals a bit more than a day
        ]

        # Start the command as a subprocess
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self):
        print(f'Saving output in {self.output}')
        # Simulate pressing a key to trigger early output
        keyboard.press_and_release('enter')  # Replace with the key that triggers early output

        # Read the output of the command as it runs
        stdout, stderr = self.process.communicate()

        # Decode stdout and stderr
        output_text = stdout.decode()

        # Regular expression pattern to match joules and seconds
        pattern = r"Energy consumption in joules: ([\d.]+) for ([\d.]+) sec"

        # Search for the pattern in the output
        match = re.search(pattern, output_text)

        self.cleanup()

        # If a match is found, extract the joules and seconds values
        if match:
            self.joules = match.group(1)
            self.seconds = match.group(2)
            return self.joules, self.seconds
        else:
            print("Energy consumption data not found.")
            return None, None

    def cleanup(self):
        print(f'Cleaning up process...')
        self.process.kill()
        if self.process.poll() is None:
            print("Terminating the process.")
            self.process.terminate()  # Gracefully terminate the process
            self.process.wait()  # Wait for the process to terminate
        else:
            print("Process already terminated.")


if __name__ == "__main__":
    # This block will only be executed when the script is run directly
    energi = EnergiCustom()
    energi.start()  # Start the subprocess
    for i in range(2):
        time.sleep(1)
        print(f'Sleeping {i}')

    joules, seconds = energi.stop()  # Stop and get joules and seconds

    if joules and seconds:
        print(f"Energy consumption: {joules} joules for {seconds} sec of execution.")
