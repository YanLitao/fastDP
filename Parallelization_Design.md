## Data and Preprocessing

### Data Description
Our data comes from American Community Survey Public Use Microdata Sample (PUMS) files. It includes useful but somehow sensitive census information such as Sex, Married, College degree. Our objective is to train a deep learning model to predict the unemployment rate based on other demographic information using DPSGD and HPC and HTC tools so that we can both protect privacy and obtain a satisfiable runtime of the algorithm.


### Sequential Version
    1. Data Balancing Figure

### Parallelization 
    1. MapReduce vs Spark
    
    2. What’s the Principle of Spark’s Parallelization? (what's the advantage of Spark over Mapreduce)

## Neural Network Model: use a big MLP

### Model Design
    1. 

### Parallelization 
    1. A Pseudocode of Parallel Version of SGD
    
    2. Model Parallel vs Data Parallel
    
    3. Parameter Server vs Tree Reduction
    
    4. From Scratch Version
    
    5. Distributed Data Parallel Package Version 

### Advanced features
    1. Ring All-Reduce

