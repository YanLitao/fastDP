# Reproduction

## Data Processing

### MapReduce

### Spark

## Distributed DPSGD with GPU acceleration

**1. Creating the Nodes**

  a. login **AWS EC2** and select **Launch Instance**.
  
  b. Choose an **Amazon Machine Image (AMI)** - Here we will select the **Deep Learning AMI (Ubuntu 16.04) Version 28.1**. 
  
  c. Choose an Instance Type - Choose **g3.4xlarge** for testing multiple nodes with 1 GPU each; choose **g3.8xlarge** for testing multiple nodes with multiple GPU;
  
  d. Configure Instance Details - We only need to increase the **number of instances** to **2**;
  
  e. Add Storage - The default setting of storage is only 75 GB. And the default storage is enough for the STL-10 dataset. But, if you want to train on a larger dataset such as ImageNet, you will have to add much more storage just to fit the dataset and any trained models you wish to save;
  
  f. Add Tags - Directly click on the next step;
  
  g. Configure Security Group - This is a critical step. By default two nodes in the same security group would not be able to communicate in the distributed training setting. Here, we want to create a new security group for the two nodes to be in. However, we cannot finish configuring in this step. For now, just remember your new security group name (e.g. launch-wizard-12) then move on to next step;
  
  h. Review Instance Launch - Here, review the instance then launch it. By default, this will automatically start initializing the two instances. You can monitor the initialization progress from the dashboard.

**2. Environment Setup**
  
  a. activate the pytorch environment: `source activate pytorch_p36`;
  
  b. Install the latest Pytorch 1.1: `conda install pytorch cudatoolkit=10.0 -c pytorch`;
  
  c. Find the name of private IP of the node by running `ifconfig` (usually `ens3`) and export it to NCLL socket: `export NCCL_SOCKET_IFNAME=ens3` (add to `.bashrc` to make this change permanent);
  
  d. Upload the scripts to each node or `git clone` from the repository;
  
  e. Also, upload the data to each node if running without NFS (Network File System) setup;
  
  f. Repeat above steps on each node.

**3. Set up NFS**

Let `master$` denote master node and `$node` denote any other node.
  
Run the following commands on master node:
  
  a. Install NFS server: `master$ sudo apt-get install nfs-kernel-server`;
  
  b. Create NFS directory: `master$ mkdir cloud`;
  
  c. Export cloud directory: by executing `master$ sudo vi /etc/exports` to open the `/etc/exports` and add `/home/ubuntu/cloud *(re,sync,no_root_squash,no_subtree_check)` to it;
  
  d. Update the changes: `master$ sudo exportfs-a`;
  
Configure the NFS client on other nodes:

  a. Install NFS client: `node$ sudo apt-get install nfs-common`;
  
  b. Create NFS directory: `node$ mkdir cloud`;
  
  c. Mount the shared directory: `node$ sudo mount -t nfs <Master Node Private IP>:/home/ubuntu/cloud /home/ubuntu/cloud`;
  
  d. Make the mount permanent (optional): add the following line `<Master Noder Private>:/home/ubuntu/cloud /home/ubuntu/cloud nfs` to `/etc/fstab` by executing `node$ sudo bi /etc/fstab`.