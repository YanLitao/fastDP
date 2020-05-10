# Discussion and Future Work

Click <a href="https://yanlitao.github.io/fastDP/">here</a> to go back to Homepage.

## Table of Contents
1. [Goals Achieved](#goals-achieved)
  * [Data Processing](#data-processing)
  * [Distributed Computing](#distributed-computing)
2. [Future Work](#future-work)
  * [Advanced All-Reduce algorithms](#advanced-all-reduce-algorithms)
  * [Ring All-Reduce](#ring-all-reduce)

## Goals Achieved

### Data Processing

The dataset we used contains around 1.2 millions records. All data preprocessing tasks are carried out on Spark. Before using Spark to process data, it took 78.52s. And after using Spark, data processing steps only took us 3.587s. The speed-up is 21.89.

### Distributed Computing

We successfully deployed DPSGD model to multiple GPU’s and carried out experiments with varying number of nodes and batch sizes. Through parallelization, we reduced the runtime from 528s (one epoch) using a single g3.4xlarge down to 170.86 (one epoch) using 1 g3.16xlarge. We also implemented a dynamic load balancer that distributes batches of deferring sizes to GPUs based on their performance at the start of each epoch. 

## Future Work

### Advanced All-Reduce algorithms

In the Model Design section, we have already implement an All-Reduce algorithm. Due to the limit of the time, we cannot implement more advance All-Reduce algorithms. So, for the future work, we want to use some other All-Reduce algorithm to further improve the performance. 

There are many other implementations of the All-Reduce algorithms. Some that try to minimize bandwidth, some others that try to minimize latency. 

![allreduce](allreduce.png) 

(Figure 1 from [[Zhao, Canny]](https://arxiv.org/abs/1312.3020))

### Ring All-Reduce

![Ring-allreduce](ring-allreduce.jpg) 

(Figure 2 from [[Edir Garcia]](https://towardsdatascience.com/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da))

Ring All-Reduce is an algorithm that instead of having a single parameter server, they pass the parameters around to each worker in a ring fashion. The ring implementation of Allreduce has two phases. The first phase, the share-reduce phase, and then a share-only phase. We want to implement this algorithm in the future, and it may remarkably improve the speed-up.
