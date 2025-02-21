import time
from energi_custom import EnergiCustom
from implementations.jaxx_imp import jax_task
from cs4575_project1.implementations.keras_imp import keras_task
from implementations.pytorch_imp import torch_task

# Sleep task for startup
def sleep():
    for i in range(3):
        time.sleep(i)
        print(f'Sleeping for {i}')

# CHOOSE TASK "keras", "pytorch" or "jax" 
task = "keras"
# Run energybridge and task
energi = EnergiCustom(output=f"results/{task}_results.csv")
energi.start()  # Start the subprocess
sleep()

# Execute task according to chosen task
match task:
    case "pytorch":
        print("Executing pytorch task...")
        torch_task()
    case "keras":
        print("Executing keras task...")
        keras_task()
    case "jax":
        print("Executing jax task...")
        jax_task()

joules, seconds = energi.stop()  # Stop and get joules and seconds, if it hangs = deadlock, just CTRL+C
if joules and seconds:
    print(f"Energy consumption: {joules} joules for {seconds} sec of execution.")
