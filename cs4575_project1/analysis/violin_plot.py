import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cs4575_project1.analysis.results_extraction import Result
from pathlib import Path

dir = 'results_mao'
frameworks = ['keras', 'torch','jax_jit']
results = []
data_time = []
data_power = []
data_energy = []
data_edp = []
data_normalised_power = []
data_normalised_energy = []
data_normalised_edp = []
for framework in frameworks:
    res = Result(framework, Path(f"../{dir}/{framework}").resolve())
    res.extract()
    results.append(res)
    data_time.append(res.time)
    data_power.append(res.power)
    data_energy.append(res.energy)
    data_edp.append(res.edp)


# Global min-max normalization
def normalize_min_max_global(data_list):
    all_values = []
    for data in data_list:
        all_values.extend(data)

    if not all_values:
        return [[] for _ in data_list]

    global_min = min(all_values)
    global_max = max(all_values)

    normalized_data = []
    for data in data_list:
        if global_max > global_min:
            normalized = [(x - global_min) / (global_max - global_min) for x in data]
        else:
            normalized = [0.5 for _ in data]
        normalized_data.append(normalized)

    return normalized_data


data_normalised_power = normalize_min_max_global(data_power)
data_normalised_energy = normalize_min_max_global(data_energy)
data_normalised_edp = normalize_min_max_global(data_edp)

save_dir = Path(f"../{dir}/plots")
save_dir.mkdir(exist_ok=True, parents=True)


def plot_box_violin(data, labels, title, y_label, filename):
    labels = ["Keras", "PyTorch", "JAX"]
    plt.figure(figsize=(10, 6))
    positions = np.arange(len(labels))

    # Violin plot
    sns.violinplot(data=data, inner=None, linewidth=1, width=0.6, cut=0)

    # Box plot
    plt.boxplot(data, positions=positions, widths=0.2, patch_artist=True,
                boxprops=dict(facecolor="lightblue", alpha=0.6),
                medianprops=dict(color="red", linewidth=1.5),
                flierprops=dict(marker='o', color='red', markersize=6))

    plt.xticks(ticks=positions, labels=labels)
    plt.title(title, fontsize=14)
    plt.xlabel('Framework', fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    plt.grid(linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    plt.savefig(save_dir / filename, dpi=300)
    plt.show()


# <<<<<<< HEAD
plot_box_violin(data_time, frameworks, "Time to complete training and evaluation by different frameworks", "Time (s)",
                "time_plot.png")
plot_box_violin(data_power, frameworks, "Average power use to complete training and evaluation by different frameworks",
                "Power (W)", "power_plot.png")
plot_box_violin(data_energy, frameworks, "Energy consumed to complete training and evaluation by different frameworks",
                "Energy (J)", "energy_plot.png")
plot_box_violin(data_edp, frameworks, "Energy-Delay Product (EDP) Across Different Frameworks",
                "J * s", "edp_plot.png")
plot_box_violin(data_normalised_power, frameworks, "Power (normalized)", "Normalized power",
                "normalized_power_plot.png")
plot_box_violin(data_normalised_energy, frameworks, "Energy (normalized)", "Normalized energy",
                "normalized_energy_plot.png")
plot_box_violin(data_normalised_edp, frameworks, "Energy-Delay Product (EDP) (normalized)", "Normalised EDP",
                "normalized_edp_plot.png")
# =======
# plot_violin(data_time, frameworks, "Time to complete 3 epochs of training by different frameworks", "Time (s)")
# plot_violin(data_power, frameworks, "Average power used to complete 3 epochs of training by different frameworks", "Power (W)")
# plot_violin(data_energy, frameworks, "Energy consumed to complete 3 epochs of training by different frameworks", "Energy (J)")
# plot_violin(data_normalised_power, frameworks, "Power (normalized)", "Normalized power")
# plot_violin(data_normalised_energy, frameworks, "Energy (normalized)", "Normalized energy")
# >>>>>>> real-data