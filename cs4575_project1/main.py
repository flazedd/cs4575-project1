import time

from cs4575_project1.implementations.pytorch_imp import torch_task
from energi_custom import EnergiCustom
from implementations.tensorflow_imp import tensor_task

energi = EnergiCustom(output="results.csv")
energi.start()  # Start the subprocess


# Insert actual task here
def task():
    for i in range(3):
        time.sleep(i)
        print(f'Sleeping for {i}')

# task()
# tensor_task()
torch_task()

joules, seconds = energi.stop()  # Stop and get joules and seconds, if it hangs = deadlock, just CTRL+C

if joules and seconds:
    print(f"Energy consumption: {joules} joules for {seconds} sec of execution.")
