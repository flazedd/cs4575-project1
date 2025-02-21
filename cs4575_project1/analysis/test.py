from results_extraction import Result
from pathlib import Path

# relative_path =
absolute_path = Path("../results/keras").resolve()

print(absolute_path)

r = Result("test", absolute_path)
r.extract()
r.print_results()