---
author: Reinier Schep, Maosheng Jiang, Razvan Loghin, Alex Zheng
title: "Tensorflow, Torch and JAX energy consumption comparison for convolutional neural networks on MNIST"
image: "../img/p1_measuring_software/gX_template/cover.png"
date: 28/02/2025
summary: |-
    Tensorflow, Torch and Jaxx energy 
    consumption comparison for convolutional neural networks on MNIST dataset.
---

## Introduction
For this experiment we aim to compare the energy consumption of Tensorflow (keras), PyTorch and JAX when training and using a Convolutional Neural Network (CNN).
We have implemented the same exact CNN architecture for each framework and then we measure energy usage of each.


## Methodology
### CNN architecture used

The CNN architecture used is shown in the figure below *insert figure*

- **Convolutional Layer (64 filters, 3x3, ReLU, same padding)** – Extracts local features while maintaining spatial dimensions.
- **Pooling Layer (2x2, stride 2)** – Reduces spatial size to retain essential features efficiently.
- **Convolutional Layer (128 filters, 5x5, ReLU, same padding)** – Captures more complex patterns with a larger receptive field.
- **Pooling Layer (2x2, stride 2)** – Further reduces spatial dimensions to improve computational efficiency.
- **Flatten Layer** – Converts multi-dimensional feature maps into a 1D vector for classification.
- **Fully Connected Layer (10 units, softmax activation)** – Produces class probabilities for final classification.



### Hardware and software setup

The experiment was conducted on a computer with the following hardware/software:
- OS: Microsoft Windows 11 Pro (10.0.26100 Build 26100)
- CPU: AMD Ryzen 5 3600 6 cores@3593Mhz, 12 logical cores
- RAM: 16GB
- GPU: NVIDIA RTX 2060 Super

Some other settings under which the experiment runs:
- All applications are closed in task manager, except an Administrator Powershell which executes the experiment
- Notifications are turned off
- A single monitor is connected
- Internet connection is disabled

### Energy measurement
Energy measurement was done by using the tool [EnergiBridge](https://github.com/tdurieux/EnergiBridge).


### Dataset
The [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) was used for training and evaluating the CNN described earlier.


### Evaluation
Before starting the experiment, the CPU is warmed up for 5 minutes to prevent cold starts which affect energy consumption.
Then, a sequence of timestamped power measurements are taken for each framework during their execution.
After the execution of a framework, an idle time of 1 minute is introduced instead of 
directly measuring the next framework to prevent trail energy usage from influencing the energy usage of the next framework to be evaluated. 
This will be done for a total of 30 iterations, each iteration the order of frameworks evaluated 
is shuffled randomly to mitigate any potential order bias.
This results in 30 .csv files for each framework which will be used for further analysis.


## Analysis
### Statistical significance
Show some violin box plots for each framework here...
Some p-values etc.
Is the data observed normal? shapiro wilk test
Effect size analysis

### Practical significance
Do our results really matter in practice?


## Discussion

## Limitations & future work

## Conclusion