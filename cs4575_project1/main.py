import time
from energi_custom import EnergiCustom
from implementations.jax_jit_imp import jax_jit_task
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
    "jax": jax_jit_task,
}
frameworks = list(frameworks_dict.keys())
dir = './results_alex'
utils.create_framework_dirs(frameworks, dir)
energi = EnergiCustom()
iterations = 32
cooldown = 60
utils.cpu_ram_warmup(duration=300)
for i in range(0,42):
    utils.print_color(f'Frameworks before {frameworks}')
    random.shuffle(frameworks)
    utils.print_color(f'Frameworks after {frameworks}')
    for framework in frameworks:
        file_output = f"{dir}/{framework}/{framework}_{i}.csv"
        utils.print_color(f'Working on {file_output} for iteration {i}')
        energi.output = file_output
        utils.print_color(f'About to start Energi measurements...')
        start_time = time.time()
        energi.start()
        utils.print_color(f'Started Energi measurements and calling task...')
        frameworks_dict[framework]()
        utils.print_color(f'Framework task completed, stopping measurements...')
        energi.stop()
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        utils.print_color(f"Elapsed time: {elapsed_time:.2f} minutes")
        utils.print_color(f'Measurements stopped for {file_output} for iteration {i}, entering cooldown of {cooldown} seconds...')
        time.sleep(cooldown) # Pause between runs to avert trail energy consumption

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


