# Experiments and Performance Results

Click <a href="https://yanlitao.github.io/fastDP/">here</a> to go back to Homepage.

## Table of Contents
1. [Metrics of Performance](#metrics-of-performance)
2. [Data Preprocessing](#data-preprocessing)
  * [Spark Data Processing](#spark-data-processing)
  * [Distributed DPSGD](#distributed-dpsgd)
3. [Distributed DPSGD](#distributed-dpsgd)
  * [Code Baseline](#code-baseline)
  * [Experiment with Different Number of GPU](#experiment-with-different-number-of-gpu)
  * [Experiment with Different Distributations of GPU](#experiment-with-different-distributations-of-gpu)
  * [Money Tradeoff](#money-tradeoff)

## Metrics of Performance

while measuring the performance of the experiment, we focus on the following metrics:

average time per epoch: each experiment we run 10 epoch for the training, and we use average execution time per epoch as the indication of speed for the experiment, since DPSGD optimization and parallelization mainly happen in the training loop of the model. 

speedup: speedup is a number that measures the relative performance of two systems processing the same problem. Specifically, if the amount of time to complete a work unit with 1 processing element is t1, and the amount of time to complete the same unit of work with N processing elements is tN, the strong scaling speedup is t1/tn. We use strong scaling speedup in the following paragraphs.

Strong scaling vs Weak scaling for measuring ML performance

## Data Preprocessing

### MapReduce Data Processing

 * Results

| Percentage of Full Dataset | Sequential Time | with 1 cores | with 2 cores | with 4 cores |
|----------------------------|-----------------|--------------|--------------|--------------|
| 1%                         | 0\.927s         | 110s         |66s           | 44s          |
| 25%                        | 17\.579s        | 114s         | 68s          | 44s          |
| 100%                       | 78\.52s         | 126s         | 76s          | 48s          |
 
### Spark Data Processing
 * Overhead Analysis

 * Results

| Percentage of Full Dataset | Sequential Time | 1 cores  | 2 cores   | 3 cores    | 4 cores    |
|----------------------------|-----------------|----------|-----------|------------|------------|
| 100%                       | 78\.52s         | 6\.3429s | 3\.61228s | 3\.587001s | 4\.185391s |

 * Strong scaling, weak scaling 

## Distributed DPSGD

### Code Baseline

Original SGD vs Original DPSGD:
Compared to the original SGD, DPSGD could achieve approximately similar testing accuracy in terms of prediction. The runtime of DPSGD is significantly slower than the original one, and the bottleneck of the algorithm is mainly the minibatch_step in backpropagation. However, the upside of using this optimizer is we could better protect the privacy of data while training the model. Here we have demonstrated that by using DPSGD, the model can better protect training data from model inversion attack1 by lowering the attacker’s accuracy.

Model inversion attack is a famous privacy attack against machine learning models. The access to a model is abused to infer information about the training data, which raised serious concerns given that training data usually contain privacy sensitive information. We use the success rate of model inversion attack as the criteria for the effectiveness of DP training.

 **Profiling Results**

| Algorithm        | Max time \(s\) | average time \(s\) | Min time \(s\) | Test Acc  | Model Inversion Attack Acc |
|------------------|----------------|--------------------|----------------|-----------|----------------------------|
| Sequential SGD   | 5\.322         | 5\.15              | 5\.104         | 0\.66     | 0\.74                      |
| Sequential DPSGD | 528            | 510                | 487            | 0\.66     | 0\.61                      |

![baseline performance](re1.png)

### Experiment with Different Number of GPU
 
 **package version:**

| \# of Nodes     | \# GPUs per Node | Max time \(s/epoch\) | Average time \(s/epoch\) | Min time \(s/epoch\) | Test Acc |
|-----------------|------------------|----------------------|--------------------------|----------------------|----------|
| 1\*g3\.4xlarge  | 1                | 619                  | 617                      | 616                  | 0\.62    |
| 1\*g3\.8xlarge  | 2                | 312                  | 310                      | 309                  | 0\.62    |
| 1\*g3\.16xlarge | 4                | 181                  | 178                      | 175                  | 0\.615   |

![different number gpu package](re2.png)

**scratch version:**

| \# of Nodes     | \# GPUs per Node | Max time \(s/epoch\) | Average time \(s/epoch\) | Min time \(s/epoch\) | Test Acc |
|-----------------|------------------|----------------------|--------------------------|----------------------|----------|
| 1\*g3\.4xlarge  | 1                | 624                  | 615                      | 611                  | 0\.647   |
| 1\*g3\.8xlarge  | 2                | 311                  | 304                      | 298                  | 0\.65    |
| 1\*g3\.16xlarge | 4                | 174                  | 173                      | 171                  | 0\.656   |

![different number gpu scratch](re3.png)

**analysis:**
Since we are using strong scaling to calculate GPU speedup, each GPU will handle a smaller part of data as the number of GPU increases. Here we can see that the time of each epoch decreases is almost proportional to the number of GPU increases. However, we haven’t achieved ideally linear speedup because of data partition and communication overheads.

### Experiment with Different Distributations of GPU
 
 **package version:**
 
 | \# of Nodes                | \# GPUs per Node | Max time \(s/epoch\) | Average time \(s/epoch\) | Min time \(s/epoch\) | Test Acc |
|----------------------------|--------------|--------------|--------------|--------------|--------------|
| 4\*g3\.4xlarge \(1 GPU\)   | 1                | 191\.8               | 190\.3                   | 189\.3               | 0\.615   |
| 2\*g3\.8xlarge \(2 GPUs\)  | 2                | 176\.8               | 173\.05                  | 174\.9               | 0\.615   |
| 1\*g3\.16xlarge \(4 GPUs\) | 4                | 181\.4               | 175\.3                   | 177\.6               | 0\.615   |

![different distributations gpu package](re4.png)

**scratch version:**

| \# of Nodes                | \# GPUs per Node | Max time \(s/epoch\) | Average time \(s/epoch\) | Min time \(s/epoch\) | Test Acc |
|----------------------------|--------------|--------------|--------------|--------------|--------------|
| 4\*g3\.4xlarge \(1 GPU\)   | 1                | 273\.9               | 269\.8                   | 265\.3               | 0\.654   |
| 2\*g3\.8xlarge \(2 GPUs\)  | 2                | 209\.8               | 205\.7                   | 204\.9               | 0\.656   |
| 1\*g3\.16xlarge \(4 GPUs\) | 4                | 174\.3               | 172\.8                   | 170\.86              | 0\.656   |

![different distributations gpu scratch](re5.png)

**analysis:**
We also tried out different distributions of GPU clusters with 4 GPUs within each. The clusters achieve approximately the same speedup, while the node with 4 GPU builtin achieves the best performance. We think this is mainly because it has only intra-node communication which is lower than internode communication overhead.

### Money Tradeoff

| Experiment      | \# GPU | Time \(s\) | price / hour | Cost   |
|-----------------|--------|------------|--------------|--------|
| 1\*g3\.4xlarge  | 1      | 6300       | 1\.14        | 1\.995 |
| 1\*g3\.8xlarge  | 2      | 3205       | 2\.28        | 2\.03  |
| 1\*g3\.16xlarge | 4      | 1813       | 4\.56        | 2\.3   |
| 4\*g3\.4xlarge  | 4      | 2144       | 4\.56        | 2\.72  |
| 2\*g3\.8xlarge  | 4      | 1937       | 4\.56        | 2\.45  |

**analysis:**
g3.4x large is the cheapest instance among all GPU’s, but it is also pretty slow compared to multi-GPU clusters.
g3.16x large is the fastest one among all distributed GPU cluster’s, as it only has intra-node communication which is lower than inter-node communication overhead.
Hence, in our case the single node GPUs are usually best money for value given the lower overhead involved. So, you can go for any single node multi GPUs for the best combination of speed and price.
