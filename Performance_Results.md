# Experiments and Performance Results

Click <a href="https://yanlitao.github.io/fastDP/">here</a> to go back to Homepage.

## Table of Contents
1. [Metrics of Performance](#metrics-of-performance)
  * [Spark Data Processing](#spark-data-processing)
  * [Distributed DPSGD](#distributed-dpsgd)

## Metrics of Performance

while measuring the performance of the experiment, we focus on the following metrics:

average time per epoch: each experiment we run 10 epoch for the training, and we use average execution time per epoch as the indication of speed for the experiment, since DPSGD optimization and parallelization mainly happen in the training loop of the model. 

speedup: speedup is a number that measures the relative performance of two systems processing the same problem. Specifically, if the amount of time to complete a work unit with 1 processing element is t1, and the amount of time to complete the same unit of work with N processing elements is tN, the strong scaling speedup is t1/tn. We use strong scaling speedup in the following paragraphs.

Strong scaling vs Weak scaling for measuring ML performance
 
### Spark Data Processing
 * Overhead Analysis

 * Results

| Percentage of Full Dataset   | Sequential Time  | Spark with 1 core | Spark with 2 cores | Spark with 3 cores |
| --------------------------   | ---------------  | ----------------- | ------------------ | ------------------ |  
| 50%                          |                  |                   |                    |                    |
| 100%                         | 78.52s           | 6.3429s           |  3.61228s          | 3.587001s          |

 * Strong scaling, weak scaling 

### Distributed DPSGD
 * Code Baseline

 * Profiling Results

| Epoch   |  Time (s)  | Test Acc |
| ------- | ---------- | -------- |
| 1       |            |          |
| 2       |            |          | 

 * Experiment with Different Distributions of GPU

| Number of Nodes| Number of GPUs per Node  | Time (s/epoch) | Speedup |
| -------------- | ------------------------ | -------------- | ------- |
| 1              |                          |                |         |
| 2              |                          |                |         |

 * Experiment with Different Problem Size
 
 * Money Tradeoff
