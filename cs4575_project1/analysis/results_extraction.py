import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats


class Result:
    def __init__(self, name="library_uesd", path_dir="path_to_directory_with_results"):
        self.power = []
        self.energy = []
        self.time = []
        self.edp = []
        self.metrics = {}
        self.name = name
        self.path_dir = Path(path_dir).resolve()

    def extract(self):
        for file_path in self.path_dir.glob("*.csv"):
            df = pd.read_csv(file_path)
            # Compute the time
            first_time_ms = df["Time"].iloc[0]
            last_time_ms = df["Time"].iloc[-1]
            time_s = (last_time_ms - first_time_ms) / 1000
            self.time.append(time_s)

            energy_values = df["CPU_ENERGY (J)"]

            for i in range(1, len(energy_values)):  # Loop from index 1 to 500
                if df["CPU_ENERGY (J)"].iloc[i] < df["CPU_ENERGY (J)"].iloc[i - 1]:
                    # raise Exception(f"Energy measurement at index {i} is smaller than the previous value.")
                    pass

            # Compute the energy
            first_energy_measurement = df["CPU_ENERGY (J)"].iloc[0]
            last_energy_measurement = df["CPU_ENERGY (J)"].iloc[-1]
            used_energy = last_energy_measurement - first_energy_measurement
            if used_energy < 0:
                # raise Exception("Used energy is negative...")
                continue
            self.energy.append(used_energy)

            # Compute the power
            used_power = used_energy / time_s
            self.power.append(used_power)

            # Compute EDP
            edp_value = used_energy * time_s
            self.edp.append(edp_value)

        self.power = self.remove_outliers(self.power)
        self.energy = self.remove_outliers(self.energy)
        self.time = self.remove_outliers(self.time)
        self.edp = self.remove_outliers(self.edp)

    def extract_metrics(self, data, type):
        if not data:
            print(f"No data available for {type}")
            return

        shapiro_stat, shapiro_p_value = stats.shapiro(data)
        mean_val = np.mean(data)
        median_val = np.median(data)
        var_val = np.var(data, ddof=1)
        std_val = np.std(data, ddof=1)
        min_val = np.min(data)
        max_val = np.max(data)

        results = {
            "Shapiro Wilk P-Value": shapiro_p_value,
            "Mean": mean_val,
            "Median": median_val,
            "Variance": var_val,
            "Standard Deviation": std_val,
            "Minimum Value": min_val,
            "Maximum Value": max_val
        }

        self.metrics[type] = results

    def compare_results(self, other):
        print(f"Comparison between libraries {self.name} and {other.name}")
        self_dict = {
            "Time": self.time,
            "Power": self.power,
            "Energy": self.energy,
            "EDP": self.edp,
            "Normalised EDP": self.normalised_edp  # Added normalized EDP for comparison
        }

        other_dict = {
            "Time": other.time,
            "Power": other.power,
            "Energy": other.energy,
            "EDP": other.edp,
            "Normalised EDP": other.normalised_edp  # Added normalized EDP for comparison
        }

        for metric, data_self in self_dict.items():
            print(f"Statistical comparsion for {metric}")
            data_other = other_dict[metric]

            if (self.metrics[metric]['Shapiro Wilk P-Value'] > 0.05
                    and other.metrics[metric]['Shapiro Wilk P-Value'] > 0.05):
                test_name = "Independent T-test"
                stat, p_value = stats.ttest_ind(data_self, data_other, equal_var=False)
            else:
                test_name = "Mann-Whitney U Test"
                stat, p_value = stats.mannwhitneyu(data_self, data_other, alternative='two-sided')

            mean_diff = self.metrics[metric]['Mean'] - other.metrics[metric]['Mean']
            median_diff = self.metrics[metric]['Median'] - other.metrics[metric]['Median']

            print(f"    Test Used: {test_name}")
            print(f"    t-test p-value: {p_value}")
            print(f"    Mean difference: {mean_diff}")
            print(f"    Median difference: {median_diff}")
        print("---")

    def remove_outliers(self, data):
        data = np.array(data)

        mean = np.mean(data)
        std_dev = np.std(data)

        diff = np.abs(data - mean)

        # Remove all data points that deviate from the mean more than 3 standard deviations
        mask = diff <= (3 * std_dev)

        # Return the data without the outliers
        return data[mask].tolist()

    def print_results(self, k=False):
        print(f"\n Results for {self.name}")

        if k:
            print(f"Time (s): {self.time}")
            print(f"Power (W): {self.power}")
            print(f"Energy (J): {self.energy}")
            print(f"Energy-Delay Product (EDP): {self.edp}")
            print(f"Normalised EDP: {self.normalised_edp}")  # Added to print normalized EDP

        if not self.metrics:
            print("No metrics computed yet.")
            return

        print("\nComputed Metrics:")
        for metric, values in self.metrics.items():
            print(f"{metric} metrics:")
            for key, value in values.items():
                print(f"   - {key}: {value}")