import subprocess
import time
import os
import keyboard
import re
from utils import print_color

class EnergiCustom:
    def __init__(self, output="results.csv"):
        self.joules = None
        self.seconds = None
        self.process = None
        self.output = output


    def start(self):
        # Ensure the output directory exists
        output_dir = os.path.dirname(self.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print_color(f"Created output directory: {output_dir}")

        print_color(f'Output will be saved in {self.output} once you call stop()')
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the energibridge executable
        energibridge_path = os.path.join(script_dir, 'energibridge_things', 'energibridge')

        # Define the command as a list
        command = [
            energibridge_path,
            '-o', self.output,
            '--summary',
            'timeout', '99999'  # Maximum for Windows, ~1 day
        ]
        # Start the command as a subprocess
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print_color(f"Failed to start energibridge: {e}", success=False)
            raise

    def stop(self):
        print_color(f'Saving output in {self.output}')
        # Simulate pressing a key to trigger early output
        keyboard.press_and_release('enter')  # Replace with the key energibridge expects
        time.sleep(5)
        self.cleanup()

        # Read the output of the command
        # stdout, stderr = self.process.communicate()
        #
        # # Decode stdout and stderr
        # output_text = stdout.decode()
        # error_text = stderr.decode()
        #
        # # Regular expression pattern to match joules and seconds
        # pattern = r"Energy consumption in joules: ([\d.]+) for ([\d.]+) sec"
        #
        # # Search for the pattern in the output
        # match = re.search(pattern, output_text)
        #
        # self.cleanup()
        #
        # # If a match is found, extract the joules and seconds values
        # if match:
        #     self.joules = match.group(1)
        #     self.seconds = match.group(2)
        #     return self.joules, self.seconds
        # else:
        #     print_color(f"Energy consumption data not found in output: {output_text}", success=False)
        #     if error_text:
        #         print_color(f"Error output: {error_text}", success=False)
        #     print_color("Possible causes: energibridge failed, RAPL not running (run 'sc start rapl' in Admin CMD), or output directory issue.", success=False)
        #     return None, None

    def cleanup(self):
        print_color(f'Cleaning up process...')
        self.process.kill()
        if self.process.poll() is None:
            print_color("Terminating the process.")
            self.process.terminate()  # Gracefully terminate
            self.process.wait()  # Wait for termination
        else:
            print_color("Process already terminated.")

if __name__ == "__main__":
    energi = EnergiCustom(output="results/test.csv")
    energi.start()  # Start the subprocess
    for i in range(2):
        time.sleep(1)
        print(f'Sleeping {i}')
    joules, seconds = energi.stop()  # Stop and get joules and seconds
    if joules and seconds:
        print(f"Energy consumption: {joules} joules for {seconds} sec of execution.")