# Tutorial - Setting Up Cloud GPU for Running CUDA.jl

Welcome to this tutorial! If you're looking to use GPUs in the cloud for running Julia programs with CUDA.jl, you're in the right place. This tutorial aims to help you set up cloud GPU on Amazon Web Service (AWS) to run CUDA.jl.

## Prerequisite

Before the tutorial begins, you should launch P type EC2 instance(s) on your own in AWS. Here are some links for you to quickly get to know key concepts and suceessfully launch your instance(s).

- What is Amazon EC2? https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html
- Which type of EC2 should I use?
  https://aws.amazon.com/ec2/instance-types/
- How to launch Amazon EC2? https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html
- What is spot instance? (optional) https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html
- How to work with spot instance? (optional) https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-requests.html

The last two resources are optional if the cost isn't a concern for you, and you can choose to stick with on-demand instance(s). However, please note that in both cases you will need to manually request a quota for P type instances before launching.

## Connect to Instance(s)

Up to this step, we assume that you have already launched your EC2 instance(s). Now, you can connect to your instance(s) via your local termial or any IDE with an SSH extension.

Move your key pair file (.pem) to the `.ssh` folder:

```
$ mv ~/Downloads/<your-key-pair-name>.pem ~/.ssh/<your-key-pair-name>.pem
```

Change the permission of the key pair file:

```
$ chmod 600 ~/.ssh/<your-key-pair-name>.pem
```

Copy the public IPv4 DNS of your instance(s) from AWS and then connect to your instance(s) via SSH:

```
$ ssh -i ~/.ssh/<your-key-pair-name>.pem ubuntu@<your-ec2-public-ipv4-dns>
```

When prompted with the question `Are you sure you want to continue connecting (yes/no)?`, enter `yes`. At this point, you should be able to successfully connect to your instance(s).

## Configure Julia Package

In this step, we assume that you have already connected to your instance(s) and the commands are being executed in your instance(s)'s terminal.

You can find the version of the Julia package you want from https://julialang.org/downloads/. For illustration, we are using Julia version 1.9.0 for Linux on x86 64-bit.

Copy the address of the Julia package and download it to your instance(s):

```
$ wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.0-linux-x86_64.tar.gz
```

Extract your downloaded package:

```
$ tar -xvf julia-1.9.0-linux-x86_64.tar.gz
```

Create a symbolic link to the Julia:

```
$ sudo ln -s ~/julia-1.9.0/bin/julia /usr/local/bin/julia
```

Remove the original Julia package (optional):

```
$ rm julia-1.9.0-linux-x86_64.tar.gz
```

Then you should be able to run `julia` command successfully in your instance(s)'s root directory.

## Configure CUDA Toolkit

Like the former step, we assume that you have connected to your instance(s) and the commands are being executed in your instance(s)'s terminal.

You can find the version of CUDA repository you want from https://developer.download.nvidia.com/compute/cuda/repos/. For illustration, we choose to use CUDA version 12.1 for Ubuntu 22.04 x86-64.

Update available packages and download `build-essential` and `dkms` packages:

```
$ sudo apt-get update
$ sudo apt-get install -y build-essential dkms
```

Download the CUDA repository public key:

```
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
```

Give the CUDA packages a higher priority:

```
$ sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
```

Fetch the GPG key for the CUDA repository and add it to the system

```
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
```

Add the CUDA repository to the system:

```
$ sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
```

Update available packages again and install CUDA toolkit:

```
$ sudo apt-get update
$ sudo apt-get install -y cuda
```

Then we have to set up environment variables for cuda in system.

Open `.bashrc` file in instance terminal:

```
$ nano .bashrc
```

Add the following at the end of `.bashrc` file:

```
export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
```

Then you should save and exit the `.bashrc` file. After opening a new terminal, you can successfully check the version of the CUDA toolkit you installed by running `nvcc --version` command.

## Add CUDA to Julia

In this step, we are going to add CUDA package to Julia. Again, this is based on the fact that you have already connected to your insrance(s).

Enter into Julia:

```
$ julia
```

Add CUDA to Julia:

```
] add CUDA
```

Then you can simply test CUDA in Julia by creating a `test.jl` file like below:

```
# This is a test.jl file
using CUDA
CUDA.versioninfo()
```

If the test result displays the version of CUDA without any errors, you have successfully added CUDA to Julia.

## Enable SSH with Git Repository (optional)

This step is optional but recommended for those who are going to operate their git repository on EC2 instance(s).

Create an SSH key in your local terminal:

```
$ ssh-keygen -t rsa -b 4096 -C <your-email-address>
```

Save the key to the default directory `~/.ssh` and enter a passphrase of your choice.

Change the permission of your SSH key:

```
$ chmod 600 ~/.ssh/id_rsa
```

Add your SSH private key to the ssh-agent:

```
$ ssh-add -k ~/.ssh/id_rsa
```

Copy the public key and use it to create a new SSH key in your GitHub.

Copy the public key into your EC2 instance(s):

```
$ cat ~/.ssh/id_rsa.pub | ssh -i ~/.ssh/<your-key-pair-name>.pem ubuntu@<your-ec2-public-ipv4-dns> "cat >> .ssh/authorized_keys"
```

Copy the private key to your EC2 instance(s):

```
$ scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/<your-key-pair-name>.pem ~/.ssh/id_rsa ubuntu@<your-ec2-public-ipv4-dns>:~/.ssh/
```

Then you can directly `git clone` and `git push` (or other operations) in your EC2 instance(s).

## In The End

After going through the previous steps, you can use CUDA.jl on your cloud GPU through AWS. Please remember to terminate your instance(s) when you no longer need them, as they will continue to incur charges.
