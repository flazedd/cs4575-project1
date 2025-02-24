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
As machine learning (ML) models grow in complexity and scale, 
their computational demands have increased, leading to increased 
energy consumption. Training large-scale deep learning models 
can require as much energy as powering multiple households for
weeks, with some studies estimating that training a single deep
learning model can emit as much carbon as five cars over their 
lifetime [Strubell et al., 2019](https://aclanthology.org/P19-1355/). Given the rapid expansion of AI applications across industries, optimizing the energy efficiency of ML frameworks is critical for reducing both operational costs and environmental impact.

This paper investigates the energy efficiency of three widely 
used ML frameworks: Keras, PyTorch, and JAX. Each of these 
frameworks offers distinct design philosophies and computational 
optimizations which may significantly impact their 
energy consumption. While extensive research has been done to compare these frameworks in terms 
of training speed and model accuracy, fewer studies have focused on their power
consumption and energy efficiency. Given the scale at which these ML libraries can be deployed in the real world, 
small differences could quickly lead to significant practical differences in energy consumption.

By systematically measuring energy usage for the same exact workload across 
different frameworks for multiple iterations, 
this study aims to provide insights into how ML engineers can make more 
sustainable choices when selecting a ML framework. The results will be valuable 
for researchers, developers, and organizations seeking to balance model 
performance with environmental responsibility and associated costs.

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
- Python 3.11.8
- Poetry 1.8.3 (dependency management)
- Tensorflow (keras) 2.18.0
- Torch 2.6.0
- Jax  0.5.0
- Other dependencies can be found in the Github repository used to carry out the [experiment](https://github.com/flazedd/cs4575-project1) 
- [EnergiBridge 0.0.7](https://github.com/tdurieux/EnergiBridge/releases/tag/v0.0.7) is used and the necessary files are already included in the repository

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
Before starting the experiment, the CPU is warmed up for 5 minutes by doing calculations to prevent cold starts which affect energy consumption.
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
We set a seed for each ML framework so that it performs the same 
computations across iterations which reduces variability. 
A limitation is that we couldn't get the frameworks to all start at the same point 
so that they would produce the same exact weights and accuracy and so on.
Still, this should not affect the energy measurements since all frameworks 
have gone through the same amount of epochs of training. 
For future work, different versions of the same framework could be 
used to examine energy efficiency differences between versions. This is relevant
because when you select a ML framework to work with, you also have to select some
version to use. The experiment could also be extended to completely 
new ML frameworks like ...?.



## Conclusion