import time
from energi_custom import EnergiCustom
from implementations.jaxx_imp import jax_task
from cs4575_project1.implementations.keras_imp import keras_task
from implementations.pytorch_imp import torch_task
import utils
import random
# Sleep task for startup
def sleep():
    for i in range(3):
        time.sleep(i)
        utils.print_color(f'Sleeping for {i}')

frameworks_dict = {
    "keras": keras_task,
    "torch": torch_task,
    "jaxx": jax_task,
}
frameworks = list(frameworks_dict.keys())
utils.create_framework_dirs(frameworks)
energi = EnergiCustom()
iterations = 2
utils.cpu_ram_warmup()
for i in range(iterations):
    random.shuffle(frameworks)
    utils.print_color(f'Randomly shuffled list order: {frameworks}')
    for framework in frameworks:
        file_output = f"results/{framework}/{framework}_{i}.csv"
        utils.print_color(f'Working on {file_output} for iteration {i}')
        energi.output = file_output
        utils.print_color(f'About to start Energi measurements...')
        energi.start()
        utils.print_color(f'Started Energi measurements and calling task...')
        frameworks_dict[framework]()
        utils.print_color(f'Framework task completed, stopping measurements...')
        energi.stop()
        utils.print_color(f'Measurements stopped for {file_output} for iteration {i}, entering cooldown...')
        time.sleep(10) # Pause between runs to avert trail energy consumption

print('')
utils.print_color('Finished generating all .csv files!')

# energi= EnergiCustom()
# energi.start()  # Start the subprocess
# sleep()
# joules, seconds = energi.stop()  # Stop and get joules and seconds, if it hangs = deadlock, just CTRL+C
# if joules and seconds:
#     utils.print_color(f"Energy consumption: {joules} joules for {seconds} sec of execution.")

# Execute task according to chosen task
# match task:
#     case "pytorch":
#         utils.print_color("Executing pytorch task...")
#         torch_task()
#     case "keras":
#         utils.print_color("Executing keras task...")
#         keras_task()
#     case "jax":
#         utils.print_color("Executing jax task...")
#         jax_task()


