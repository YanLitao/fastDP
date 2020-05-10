# Discussion

Click <a href="https://yanlitao.github.io/fastDP/">here</a> to go back to Homepage.

## Table of Contents
1. [Goals Achieved](#goals-achieved)
2. [Future Work](#future-work)
  * [Ring All-Reduce](#ring-all-reduce)

## Goals Achieved

## Future Work

### Advanced All-Reduce

**1. Advanced All-Reduce algorithms**

In the Model Design section, we have already implement an All-Reduce algorithm. Due to the limit of the time, we cannot implement more advance All-Reduce algorithms. So, for the future work, we want to use some other All-Reduce algorithm to further imporve the performance. 

There are many other implementations of the Allreduce algorithm. Some that try to minimize bandwidth, some others that try to minimize latency. 

![allreduce](allreduce.png) (Figure 1 from [[Zhao, Canny]](https://arxiv.org/abs/1312.3020))

**2. Ring All-Reduce**

![Ring-allreduce](ring-allreduce.jpg) (Figure 2 from [[Edir Garcia]](https://towardsdatascience.com/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da))

Ring All-Reduce is an algorithm that instead of having a single parameter server, they pass the parameters around to each worker in a ring fashion. The ring implementation of Allreduce has two phases. The first phase, the share-reduce phase, and then a share-only phase. We want to implement this algorithm in the future, and it may remarkably imporve the speed-up.
