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
        self.normalised_power = []
        self.normalised_energy = []
        self.min_power = float('inf')
        self.max_power = float('-inf')
        self.min_energy = float('inf')
        self.max_energy = float('-inf')
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

            # Compute the energy
            first_energy_measurement = df["CPU_ENERGY (J)"].iloc[0]
            last_energy_measurement = df["CPU_ENERGY (J)"].iloc[-1]
            used_energy = last_energy_measurement - first_energy_measurement
            self.energy.append(used_energy)
            self.max_energy = max(self.max_energy, used_energy)
            self.min_energy = min(self.min_energy, used_energy)

            # Compute the power
            used_power = used_energy / time_s
            self.power.append(used_power)
            self.max_power = max(self.max_power, used_power)
            self.min_power = min(self.min_power, used_power)

        for e in self.energy:
            norm_e = e - self.min_energy / (self.max_energy - self.min_energy)
            self.normalised_energy.append(norm_e)

        for p in self.power:
            norm_p = p - self.min_power / (self.max_power - self.min_power)
            self.normalised_power.append(norm_p)

    def extract_metrics(self, data, type):
        shapiro_p_value = stats.shapiro(data)
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
            "time": self.time,
            "power": self.power,
            "energy": self.energy
        }

        other_dict = {
            "time": other.time,
            "power": other.power,
            "energy": other.energy
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
            median_diff = self.metrics[metric]['Median'] - self.metrics[metric]['Median']

            print(f"t-test p-value: {p_value}")
            print(f"Mean difference: {mean_diff}")
            print(f"Median difference: {median_diff}")
        print("---")

    def print_results(self):

        print(f"\nResults for {self.name}")
        print(f"Time (s): {self.time}")
        print(f"Power (W): {self.power}")
        print(f"Energy (J): {self.energy}")

        if not self.metrics:
            print("No metrics computed yet.")
            return

        for metric, values in self.metrics.items():
            print(f"\n {metric} metrics:")
            for key, value in values.items():
                print(f"   {key}: {value}")