from results_extraction import Result
from pathlib import Path

#TODO: Add the actual paths to the results from each library
results_jaxx = Result("Jaxx", Path("../results/jaxx").resolve())
results_pytorch = Result("Pytorch", Path("../results/torch").resolve())
results_tensorflow = Result("Tensorflow", Path("../results/keras").resolve())

results_jaxx.extract()
results_pytorch.extract()
results_tensorflow.extract()

results_jaxx.extract_metrics(results_jaxx.time, "Time")
results_jaxx.extract_metrics(results_jaxx.energy, "Energy")
results_jaxx.extract_metrics(results_jaxx.power, "Power")

results_pytorch.extract_metrics(results_pytorch.time, "Time")
results_pytorch.extract_metrics(results_pytorch.energy, "Energy")
results_pytorch.extract_metrics(results_pytorch.power, "Power")

results_tensorflow.extract_metrics(results_tensorflow.time, "Time")
results_tensorflow.extract_metrics(results_tensorflow.energy, "Energy")
results_tensorflow.extract_metrics(results_tensorflow.power, "Power")

results = [results_jaxx, results_pytorch, results_tensorflow]
# Statisitical Comparison
results_jaxx.compare_results(results_pytorch)
results_jaxx.compare_results(results_tensorflow)
results_pytorch.compare_results(results_tensorflow)

# Visualisation
