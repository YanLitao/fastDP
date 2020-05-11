# Experiments and Performance Results

Click <a href="https://yanlitao.github.io/fastDP/">here</a> to go back to Homepage.

## Table of Contents
1. [Metrics of Performance](#metrics-of-performance)
  * [Spark Data Processing](#spark-data-processing)
  * [Distributed DPSGD](#distributed-dpsgd)

## Metrics of Performance
 
### Spark Data Processing
 * Overhead Analysis

 * Results

| Percentage of Full Dataset   | Sequential Time  | Spark with 1 core | Spark with 2 cores | Spark with 4 cores |
| --------------------------   | ---------------  | ----------------- | ------------------ | ------------------ |  
| 50%                          |                  |                   |                    |                    |
| 100%                         |                  |                   |                    |                    |

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
