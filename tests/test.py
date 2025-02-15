import pandas as pd


# Function to compute total energy and total time
def compute_energy_and_time(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Assuming 'CORE0_ENERGY (J)' and 'CPU_ENERGY (J)' are columns with energy data
    total_energy_used = df['CORE0_ENERGY (J)'].sum() + df['CPU_ENERGY (J)'].sum()

    # Assuming 'Delta' and 'Time' are separate columns for time (in nanoseconds or other small units)
    total_time = df['Delta'].sum() + df['Time'].sum()

    # Convert energy to joules if needed (assuming they are in large scientific notation values already)
    # Convert time from units (likely in nanoseconds) to seconds if necessary
    total_energy_used /= 1e12  # Convert from (Joules * 10^12) to joules (if in large values)
    total_time /= 1e9  # Convert from nanoseconds to seconds

    # Print the results
    print(f"Total Energy Used: {total_energy_used:.2f} Joules")
    print(f"Total Time: {total_time:.4f} seconds")


# Provide the path to your results.csv file
file_path = 'results.csv'

# Compute and print the total energy and time
compute_energy_and_time(file_path)
