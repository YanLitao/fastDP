# Discussion

Click <a href="https://yanlitao.github.io/fastDP/">here</a> to go back to Homepage.

## Table of Contents
1. [Goals Achieved](#goals-achieved)
2. [Future Work](#future-work)
  * [Ring All-Reduce](#ring-all-reduce)

## Goals Achieved

## Future Work

### Ring All-Reduce

Ring All-Reduce is an algorithm that instead of having a single parameter server, they pass the parameters around to each worker in a ring fashion. The ring implementation of Allreduce has two phases. The first phase, the share-reduce phase, and then a share-only phase. We want to implement this algorithm in the future, and it may remarkably imporve the speed-up.
