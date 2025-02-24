import os
from colorama import Fore, Style
import numpy as np
import time
from datetime import datetime

def create_framework_dirs(frameworks, base_dir="./results"):
    """
    Creates directories for each framework inside the specified base directory if they don't already exist.

    Parameters:
    - frameworks (list): A list of framework names.
    - base_dir (str): The base directory where framework folders will be created. Default is "./results".
    """
    os.makedirs(base_dir, exist_ok=True)  # Ensure the base directory exists

    for framework in frameworks:
        framework_dir = os.path.join(base_dir, framework)
        os.makedirs(framework_dir, exist_ok=True)  # Create directory if it doesn't exist
        print(f"Directory created (or already exists): {framework_dir}")


def print_color(message, success=True):
    timestamp = datetime.now().strftime("[%H:%M]")  # Get current time in [HH:MM] format
    prefix = f"{Fore.GREEN}[+]{Style.RESET_ALL}" if success else f"{Fore.RED}[-]{Style.RESET_ALL}"
    print(f"{timestamp} {prefix} {message}")


def cpu_ram_warmup(duration=10, matrix_size=5000):
    """
    Warms up CPU and RAM by performing matrix multiplications for a given duration.

    Parameters:
    - duration (int): Time in seconds to run the warm-up.
    - matrix_size (int): Size of the random matrices for computation.
    """
    print_color(f"Warming up CPU and RAM for {duration} seconds...")

    # Allocate a large matrix to warm up RAM
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)

    start_time = time.time()

    # Perform matrix multiplication to keep CPU busy for the given duration
    while time.time() - start_time < duration:
        np.dot(A, B)  # Matrix multiplication (CPU-intensive)

    print_color("Warm-up complete.")
