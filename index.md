---
title: Differentiable Private Stochastic Gradient Descent
---

# Introduction

## Motivation and Project Statement
Differential privacy is a framework for measuring the privacy guarantees provided by an algorithm. We can design differentially private machine learning algorithms to train on sensitive data. It provides provable guarantees of privacy (point to first image), reducing the risk of exposing sensitive training data through the learned classifier. Intuitively, for any two adjacent training sets that are only differed by one data record, the learned classifiers should not be too different. In the context of deep learning, differential private stochastic gradient descent, i.e. DPSGD, is the state-of-art algorithm to train such a privacy-preserving neural network.

## The Need for Big Data and HPC
Nowadays, the DPSGD algorithm is in urgent need of combining with big compute and big data technology. On the one hand, due to the features of the DPSGD algorithm, such as limiting the gradient size in each step of parameter update, its convergence time will be 10 to 100 times longer than that of the original SGD algorithm. Without the use of computational processing, the training process of DPSGD will be extremely time-consuming. On the other hand, the datasets processed by DPSGD will be up to thousands of petabyte. For example, in some high-tech companies, they need to process billions of private user data every day, and it is impossible to process data without big data processing. Therefore, we aim to improve the performance of DPSGD algorithm by using big compute and big data technology, so that DPSGD can be applied to more complex application scenario and work with huge amount of sensitive data. 

## Our Solution

### DPSGD Algorithm
As an optimization method, differential private SGD is developed from original SGD. In the process of parameter updating, there are two more steps for DPSGD in each iteration: gradient clipping and noise addition. These two steps reduce the effect of one single anomaly so that the results of two similar datasets will not be too different. We can deploy parallel computing methods in this parameter  updating part.

### Dataset
Our data comes from American Community Survey Public Use Microdata Sample (PUMS) files. It includes useful but somehow sensitive census information such as Sex, Married, College degree. Our objective is to train a deep learning model to predict the unemployment rate based on other demographic information using DPSGD and HPC and HTC tools so that we can both protect privacy and obtain a satisfiable runtime of the algorithm.

### Solutions
Our program separates into two stages, data preprocessing and the DPSGD training. Accordingly, the levels of parallelism we are going to implement are big data and distributed parallelization technology. To be more specific, within the data preprocessing stage, we will use Spark to process large amount of data. This is because in our pilot studies, Spark runs much faster than MapReduce. And, for DPSGD training, we will use distributed package of PyTorch with MPI backend to implement the parallelized version of the parameter update iteration, which involves gradient calculation, clipping and noise addition. 

We find that there are three popular ways to implement distributed version of stochastic gradient descent: synchronous fashion, asynchronous fashion, and ring all-reduce; we plan to extend these algorithms to design distributed version of DPSGD, either centralized or decentralized. In terms of the infrastructure, since it is hard for AWS to approve our request of more than 5 g3.4xlarge instances, we currently plan to use 4 g3.4xlarge instances to run the distributed version of DPSGD, but it might change later if we are able to request more nodes. 

## Table of Contents

- [Main Features](http://YanLitao.github.io/fastDP/MF)
