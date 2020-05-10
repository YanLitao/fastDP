# Model Design

Click <a href="https://yanlitao.github.io/fastDP/">here</a> to go back to Homepage.

## Table of Contents
1. [Data and Preprocessing](#data-and-preprocessing)
  * [Data Description](#data-description)
  * [Sequential Version](#sequential-version)
  * [Parallelization](#parallelization)
2. [Neural Network Model: use a big MLP](#neural-network-model-use-a-big-mlp)
  * [Network Design](#network-design)
  * [Differential Private Stochastic Gradient Descent](#differential-private-stochastic-gradient-descent)
  * [Parallelization Design](#parallelization-design)

## Data and Preprocessing

### Data Description
Our data comes from American Community Survey Public Use Microdata Sample (PUMS) files. It includes useful but somehow sensitive census information such as Sex, Married, College degree. Our objective is to train a deep learning model to predict the unemployment rate based on other demographic information using DPSGD and HPC and HTC tools so that we can both protect privacy and obtain a satisfiable runtime of the algorithm.


### Sequential Version
    1. Data Balancing Figure

### Parallelization 
    1. MapReduce vs Spark
    
    2. What’s the Principle of Spark’s Parallelization? (what's the advantage of Spark over Mapreduce)


## Model Training

### Neural Network Architecture
Since we are facing a classification problem, and we are working on a dataset with tabular format where there are no explicit correlations between columns, we choose multilayer perceptron (MLP) as our differentially private model to predict the unemployment rate. Hyperparameters such as number of layers and hidden dimensions are tuned on our dataset, with careful consideration for the tradeoff between model performance and total running time.


### Differential Private Stochastic Gradient Descent
Differential privacy is a framework for measuring the privacy guarantees provided by an algorithm. Through the lens of differential privacy, we can design machine learning algorithms that responsibly train models on private data. 
To train a differentially private model, we rely on Differentially Private Stochastic Gradient Descent (DPSGD) [[Abadi et al.]](https://arxiv.org/abs/1607.00133).  

As an optimization method, differential private SGD is developed from vanilla stochastic gradient descent, which is the basis for many optimizers that are popular in machine learning. There are two modifications needed to ensure that stochastic gradient descent is a differentially private algorithm:

- First, the sensitivity of each gradient needs to be bounded. In other words, we need to limit how much each individual training point can influence the gradient computation. This can be done by simply clipping each gradient computed on each training point. 

- Second, we need to randomize the algorithm’s behavior to make it statistically impossible to know whether or not a particular training data was included in the training set by comparing the gradient updates. This is achieved by sampling random Gaussian noise and adding it to the clipped gradients.

![dpsgd](dpsgd.png)

### Parallelization Design

#### Parallelization Design Choices Review   
There has been quite a bit of work on parallel machine learning approaches In this section, we review some design choices in distributed deep learning training. 

- Model Parallel vs Data Parallel  
    
    There are two approaches to parallelize the training of neural networks: *model parallelism* and *data parallelism*. Model parallel "breaks" the neural network into different parts and place different parts on different nodes. For instance, we could put the first half of the layers on one GPU, and the other half on a second one. However, this approach is rarely used in practice because of the huge communication and scheduling overhead [[Mao]](https://leimao.github.io/blog/Data-Parallelism-vs-Model-Paralelism/).
    
    Data parallelization divides the dataset across all available GPU per nodes, and each process holds a copy of the current neural network, called *replica*. Each node computes gradients on its own data, and they merge the gradients to update the model parameters. Different ways of merging gradients lead to different algorithms and performance [[Arnold]](http://seba1511.net/dist_blog/article.pdf). 
    
- Parameter Server vs AllReduce
    
    There are two options to setup the architecture of the system: *parameter server* and *tree-reductions*. For the case of parameter server, one machine is responsible for holding and serving the global parameters to all replicas, which serves as a higher-level manager process. However, as discussed in [[Arnold]](http://seba1511.net/dist_blog/article.pdf), parameter servers tend to have worse scalability than tree-reduction architectures. 
    
    Allreduce is an MPI-primitive which allows normal sequential code to work in parallel, implying very low programming overhead. This allows gradient aggregation and parameter updating. The massage-passing interface (MPI) and its collective communcation operations (e.g. scatter, gather, reduce) are used to implement AllReduce algorithm. The fundamental drawbacks are poor performance under misbalanced loads and difficulty with models that exceed working memory in size. There are many implementations of the Allreduce algorithm, as shown below.
    
    ![allreduce](allreduce.png) (Figure 2 from [[Zhao, Canny]](https://arxiv.org/abs/1312.3020))
    
- CPU vs GPU

	GPU(Graphics Processing Unit) is considered as heart of Deep Learning. While CPUs can run the operating system and perform traditional serial or multi-threading tasks, GPUs have strong vector processing capabilities that enable them to perform parallel operations on very large sets of data. GPU-accelerated computing is one kind of Heterogeneous Computing, which makes use of GPU together with a CPU to accelerate deep learning. 
    
In this project, we use **data parallelism** with **AllReduce appoach** and **GPU-accelerated computing** to implement distributed version of DPSGD. 


#### Distributed Data Parallel Package  

We first implement a version of distributed DPSGD using PyTorch Distributed Data Parallel module with CUDA. 
Distributed Data Parallel module is a well-tested and well-optimized version for multi-GPU distributed training. When we wrap up our model with DistributedDataParallel, the constructor of DistributedDataParallel will register the additional gradient reduction functions on all the parameters of the model at the time of construction so that we do not need to explicitly handle gradient aggregation and parameter updates across the computational nodes during the model training.

```python
dp_device_ids = [local_rank]
device = torch.device('cuda', local_rank)
model = Network()
model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)
```

 Besides that, we use PyTorch Distributed Sampler module to implement a data sampler to automatically distribute data batch instead of hand-engineer data partition. 

    
#### From Scratch Version  

We directly implement synchronous distributed from scratch DPSGD through PyTorch distributed package module for multi-GPU communication. In this algorithm, all replicas average all of their gradients at every batch of data. Suppose the batch size for each replica is *B*, and the total number of replicas is *R*, then the **overall batchsize** is *BR*.
The following pseudo-code describes synchronous distributed DPSGD at the replica-level, for *R* replicas, *T* iteration steps, and *B* individual batch size.  

![Distributed DPSGD](distdpsgd.png)  

There are two main differences compared with sequential version of DPSGD: *data partition* and *gradient AllReduce*. For data partition stage, we divide the dataset into different pieces and assign each node one of the pieces. Later in the model training stage, each node will only sample batch data from its own portion of the data. This avoids the need of communicating the split of data across each node during the training stage. During the forward and backward propagation, each GPU calculates its own loss, calcualte and process the corresponding gradient which involves clipping and noise addition. All-Reduce is a combined operation of reduce and broadcast in MPI. In the *gradient AllReduce* step, the all local gradients are averaged (reduction) and are used to update model parameters across all of the devices (broadcast). 

Pytorch Distributed package is abstract and can be built on different backends. Our choice including Gloo and NCCL. However, since we are mainly working with CUDA tensors, and the collective operations for CUDA tensors provided by Gloo is not as optimized as the ones provided by the NCCL backend, we decide to use NCCL backend through out all of the experiments. 

4. Infrastructure Choices  

AWS has been used for its flexibility to customize with different environments. Since AWS G3 instances are relatively more cost-effective than P2 type instances, we choose mostly G3 for our experiments. G3 instances are back up with NVIDIA Tesla M60 GPUs, where each GPU delivering up to 2,048 parallel processing cores and 8 GiB of GPU memory.   

To save cost on storage and to prevent from downloading multiple copies of the data, we share the data folder through Network File System (NFS).


