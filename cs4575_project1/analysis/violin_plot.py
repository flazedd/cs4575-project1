import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cs4575_project1.analysis.results_extraction import Result
from pathlib import Path

frameworks = ['keras', 'torch', 'jaxx']
results = []
data_time = []
data_power = []
data_energy = []
data_normalised_power = []
data_normalised_energy = []
for framework in frameworks:
    res = Result(framework, Path(f"../results/{framework}").resolve())
    res.extract()
    results.append(res)
    data_time.append(res.time)
    data_power.append(res.power)
    data_energy.append(res.energy)
    data_normalised_power.append(res.normalised_power)
    data_normalised_energy.append(res.normalised_energy)

def plot_violin(data, labels, title, y_label):
    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data)

    # Set plot labels and title
    plt.xticks(ticks=np.arange(3), labels=labels)
    plt.title(title, fontsize=14)
    plt.xlabel('Framework', fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()

plot_violin(data_time, frameworks, "Time to complete training and evaluation by different frameworks", "Time (s)")
plot_violin(data_power, frameworks, "Average power use to complete training and evaluation by different frameworks", "Power (W)")
plot_violin(data_energy, frameworks, "Energy consumed to complete training and evaluation by different frameworks", "Energy (J)")
plot_violin(data_normalised_power, frameworks, "Power (normalized)", "Normalized power")
plot_violin(data_normalised_energy, frameworks, "Energy (normalized)", "Normalized energy")
