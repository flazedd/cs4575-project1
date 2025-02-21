from results_extraction import Result
from pathlib import Path

relative_path = "cs4575_project1/results/dummy"
absolute_path = Path(relative_path).resolve()

r = Result("test", '/Users/razvanloghin/Desktop/TUD-Shit/Y4/Q3/Sustainable SE/cs4575-project1/cs4575_project1/results/dummy')
r.extract()
r.print_results()