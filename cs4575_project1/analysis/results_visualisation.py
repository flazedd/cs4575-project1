from cs4575_project1.analysis.results_extraction import Result
from pathlib import Path

frameworks = ['keras', 'torch', 'jax', 'jax_jit']
results = []

for framework in frameworks:
    res = Result(framework, Path(f"../results_mao/{framework}").resolve())
    res.extract()
    res.extract_metrics(res.time, "Time")
    res.extract_metrics(res.energy, "Energy")
    res.extract_metrics(res.power, "Power")
    res.extract_metrics(res.edp, "EDP")
    results.append(res)
    res.print_results()

for i in range(len(results)):
    for j in range(i + 1, len(results)):
        results[i].compare_results(results[j])
