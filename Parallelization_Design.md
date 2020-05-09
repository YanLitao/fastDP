## Data and Preprocessing

### Data Description
Our data comes from American Community Survey Public Use Microdata Sample (PUMS) files. It includes useful but somehow sensitive census information such as Sex, Married, College degree. Our objective is to train a deep learning model to predict the unemployment rate based on other demographic information using DPSGD and HPC and HTC tools so that we can both protect privacy and obtain a satisfiable runtime of the algorithm.


### Sequential Version
    1. Data Balancing Figure

### Parallelization 
    1. MapReduce vs Spark
    
    2. What’s the Principle of Spark’s Parallelization? (what's the advantage of Spark over Mapreduce)


## Neural Network Model: use a big MLP

### Network Design
Since we are facing a classification problem, and we are working on a dataset with tabular format where there are no explicit correlations between columns, we choose multilayer perceptron (MLP) as our differentially private model to predict the unemployment rate. We use leakyReLU as the activation function for all but the output layer to prevent gradient vanishing. Hyperparameters such as number of layers and hidden dimensions are tuned on our dataset, with careful consideration for the tradeoff between model performance and compuational time.

### Differential Private Stochastic Gradient Descent


### Parallelization Design

1. Existing parallelization method review (change a name?):   
    1. Model Parallel vs Data Parallel  
    There are two approaches to parallelize the training of neural networks: *model parallelization* and *data parallelization*. Model parallel "breaks" the neural network into different parts and place different parts on different nodes. For instance, we could put the first half of the layers on one GPU, and the other half on a second one. However, this approach is rarely used in practice because of the huge communication and scheduling overhead \cite[].  
    Data parallelization divides the dataset across all available computational nodes, and each process holds a copy of the current neural network, called *replica*. Each node computes gradients on its own data, and they merge the gradients to update the model parameters. Different ways of merging gradients lead to different algorithms and performance. 
    
    3. Parameter Server vs Tree Reduction  
    There are two options to setup the architecture of the system: *parameter server* and *tree-reductions*. For the case of parameter server, one machine is responsible for holding and serving the global parameters to all replicas, which serves as a higher-level manager process. However, as discussed in \cite[], parameter servers tend to have worse scalability than tree-reduction architectures. Tree-reduction refers to the case when an infrastructure whose collective operations are executed without a higher-level manager process. The massage-passing interface (MPI) and its collective communcation operations (e.g. scatter, gather, reduce) are typical examples. 
    
2. From Scratch Version  
We directly implement synchronous distributed from scratch DPSGD through PyTorch distributed package module for multi-GPU communication. In this algorithm, all replicas average all of their gradients at every batch of data. Suppose the batch size for each replica is *B*, and the total number of replicas is *R*, then the **overall batchsize** is *BR*.
The following pseudo-code describes synchronous distributed DPSGD at the replica-level, for *R* replicas, *T* iteration steps, and *B* individual batch size.  
(PUT A PSEUDOCODE FOR DISTRIBUTED DPSGD HERE)  
There are two main differences compared with sequential version of DPSGD: *data partition* and *gradient AllReduce*. For data partition stage, we divide the dataset into different pieces and assign each node one of the pieces. Later in the model training stage, each node will only sample batch data from its own portion of the data. This avoids the need of communicating the split of data across each node during the training stage. During the forward and backward propagation, each GPU calculates its own loss, calcualte and process the corresponding gradient which involves clipping and noise addition. In the *gradient AllReduce* step, the all local gradients are averaged and being used to update models across all of the devices.  
Pytorch Distributed package is abstract and can be built on different backends. Our choice including Gloo and NCCL. However, since we are mainly working with CUDA tensors, and the collective operations for CUDA tensors provided by Gloo is not as optimized as the ones provided by the NCCL backend, we decide to use NCCL backend through out all of the experiments. 
    
3. Distributed Data Parallel Package  
Since we find From Scratch version of parallelization did not gain the speed up we expected, we decide to also implement distributed parallelization of DPSGD through PyTorch Distributed Data Parallel module. Distributed Data Parallel module is a well-tested and well-optimized version for multi-GPU distributed training. Besides that, we use PyTorch Distributed Sampler module to implement a data sampler to automatically distribute data batch instead of hand-engineer data partition. Gradient AllReduce step is also handled by Distributed Data Parallel module so we do not have to explicitly average the gradients in the training step.  

4. Infrastructure Choices  
AWS has been used for its flexibility to customize with different environments. Since AWS G3 instances are relatively more cost-effective than P2 type instances, we choose mostly G3 for our experiments. G3 instances are back up with NVIDIA Tesla M60 GPUs, where each GPU delivering up to 2,048 parallel processing cores and 8 GiB of GPU memory.   
To save cost on storage and to prevent from downloading multiple copies of the data, we share the data folder through Network File System (NFS).


