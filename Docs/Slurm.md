Using Slurm
To submit jobs to CS SLURM , you need to access the system through the following login node: 

            slurm-client.cs.tau.ac.il

This login node is your starting point for submitting jobs, checking job status, and managing your SLURM tasks

Requirements for Access:
You must be part of cs slurm groups to access the resources. Fill this form here. If you’re studying a relevant course – your supervising TA will make a request to add you to the right partition .
University Credentials: Use your TAU username and password to log in.
 

To access the system use ssh, use the following example:

 

#Replace 'your_username' with your actual TAU username
ssh your_username@slurm-client.cs.tau.ac.il
Your connection will be automatically routed to one of the clients nodes: c-001.cs.tau.ac.il -c-010.cs.tau.ac.il
Note: CS servers are not accessible from outside the University, so if you’re working outside the campus you will need to connect to the TAU network via the University VPN as described in the following link: https://computing.tau.ac.il/helpdesk/remote-access/communication/vpn
Warning or Errors related with Host Key - Removed/Rename "/.ssh/known_hosts"  file 

(Go to top)

Check your Resources
Slurm has various mechanisms for prioritizing resource allocation. One of these mechanisms is a partition system which prioritizes certain jobs over others on select resources.

To check partitions available to your user 

#Replace 'your_username' with your actual TAU username
sacctmgr -P -i show user -s your_username
You will get a list of partitions and accounts attach to it( You can use -p for full column with). If you will like to use a partition that is NOT in your default account you MUST use --account (see example below)

      User   Def Acct     Admin    Cluster    Account  Partition     
---------- ---------- --------- ---------- ---------- ---------- --------- -
   raanane gpu-students      None    cluster gpu-students  studentrun        
   raanane gpu-students      None    cluster gpu-students  studentba+        
   raanane gpu-students     None    cluster  gpu-research     killable               
In the example above user raanane must use --account to use partition 'killable'
 
To check the partitions available to your group

sinfo 
Output example:

PARTITION    AVAIL  TIMELIMIT  NODES  STATE NODELIST
studentrun      up    8:00:00      1   idle savant
studentbatch    up 3-00:00:00      1   idle savant
 
To check resources GPUS/CPU/Memory

sinfo -o "%20N  %10c  %10m  %25f  %30G " 
(Go to top)

Initial Priority at Submit time
When you submit a job on the CS Slurm Cluster, it gets an initial priority. The job's priority at any given time will be a weighted sum of multiple factors we have enabled (are still in fine tuning).

#Check queue - list pending jobs and job priority
squeue -t pending  --format="%.7i %.15Q %.20P %.12u %.20V %.7n %.20R"
The most important factors (high weights) are:
--partition : Is an amount of nodes (or/and amount of GPU’s) used to enclose the resources a specific lab or researcher has funded. The killable partition is the default, and it is not associated with any lab or researcher. If a specific partition (not killable) is request and there are free resources or resources which are currently used by the killable partition, jobs running in killable partition will be stopped and requeue in order to release these resources.
Fair Share Factor: This is a score each user is given, which indicates his/hers current priority potential. It is based on the portion of the computing resources that have been allocated and the resources jobs have already consumed on a monthly basis. Each user has a ‘billing account’ that registers how many resources have been allocated and used by the user. When a user submits a new job, Slurm will give a lower/higher priority based on the user's past usage. THE MORE YOU TAKE, THE LESS YOU WILL GET FairShare (FS).
 
Other factors are:
 
Job size factor: The job size factor correlates to the number of nodes or CPUs the job has requested. The higher the amount of requested CPUs the higher the JobSizeFactor. When there are two jobs with the same amount of requested CPUs, the one with shorter TimeLimit will have higher JobSizeFactor.
 
Age Factor: The age factor represents the length of time a job has been sitting in the queue and eligible to run. In general, the longer a job waits in the queue, the larger its age factor grows. 
 
TRES Factor: Different weights setting the degree that each TRES Type contributes to the job's priority. Current TRES Types are: CPU, GRES (GPU), Filesystem, Memory, Node, etc. See https://slurm.schedmd.com/tres.html
 
(Go to top)
How does Slurm decide what job to start next?
When there are free nodes, an approximate model of SLURM's behavior is this:

Step 1: Can the job in position one start now?
Step 2: If it can, remove it from the queue, start it, and continue with step 1.
Step 3: If it cannot, look at next job.
Step 4: Can it start now, without risking that the jobs before it in the queue get a higher START_TIME approximation?
Step 5: If it can, remove it from the queue, start it, recalculate what nodes are free, look at next job and continue with step 4.
Step 6: If it cannot, look at next job, and continue with step 4.
As soon as a new job is submitted and as soon as a job finishes, SLURM restarts with step 1, so most of the time only jobs at the top of the queue are tested for the possibility to start. As a side effect of this restart behavior, START_TIME approximations are normally NOT CALCULATED FOR ALL JOBS.

(Go to top)

Useful commands
sinfo #  show all available partitions and nodes

squeue # view the queue
squeue --me # shows only your jobs
squeue documentaions: https://slurm.schedmd.com/squeue.html
scancel <jobid> # cancel a job
scancel documentaions: https://slurm.schedmd.com/scancel.html  
sacct -l -j <jobid> # List accounting info about a job
 * You can find all information and options for using each command by running 'man <cmd>' or  '<cmd> --help' to see the command manual, e.g. 'man sinfo' or 'sinfo --help'

(Go to top)

Launching Jobs
The command "sbatch" should be the default command for running jobs. "srun"  can be use ONLY in specifics partitions and only for testing and develop. 

With sbatch, you submit your job and it is handled by Slurm ; you can disconnect, kill your terminal, etc. with no consequence. Your job is no longer linked to a running process. Also failures involving sbatch jobs typically result in the job being requeued and executed again

The srun command is designed for interactive use, with someone monitoring the output. The output of the application is seen as output of the srun command, typically at the user's terminal. Failures involving srun typically result in an error message being generated with the expectation that the user will respond in an appropriate fashion - slurm session WILL NOT stop and resources WILL NOT be released. 

Basic options available to the sbatch command in order to request the correct allocation of resources for your jobs (can be used in a script or at the command line):

Meaning:	Option:
Partition name (MANDATORY)	--partition
Job name (preferrably one that's easy to identify/manage)	--job-name
Redirect stdout instead of slurm-%j.out in the current directory	--output
Redirect stderr instead of job output file (see -o above)	--error
Maximum duration (in minutes). Default based on the partition	--time
How to end job when time's up	--signal
Number of cluster servers to be used	--nodes
Number of processes	--ntasks
CPU cores per process	--cpus-per-task
CPU memory (in MB)	--mem
Ask for X number of GPUs. Also, if you combine this with an -N option, you will get X of GPUs per node which you asked for with -N, not X GPUs total. SLURM does not support having varying number of GPUs per node in a job yet.	--gres=gpu:x
Nodes have features assigned to them . Users can specify which of these features are required by their job using the constraint option.

Supported Features: tesla_v100, quadro_rtx_8000, geforce_rtx_3090, titan_xp, geforce_rtx_2080,a100,a5000,a6000,l40s

Example: --constraint=" tesla_v100|quadro_rtx_8000" indicates that the job requires gpu server with features tesla_v100 OR quadro_rtx_8000

--constraint
 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

For more info: https://slurm.schedmd.com/sbatch.html

 

Below is an example of how to run a simple batch job with minimum allocation of resources (1 node + 1 GPU):

 

Write python script - awesome.py:

# Author: Cs System Example
# Name: awesome.py

print('hello awesome world')
 

Write submit file - awesome.slurm:

#! /bin/sh

#SBATCH --job-name=awesome
#SBATCH --output=<your_dir>/awesome.out # redirect stdout
#SBATCH --error=<your_dir>/awesome.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)
#SBATCH --time=1 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


python awesome.py
 

Submit job:

$ sbatch awesome.slurm
Submitted batch job 214726
A script can also be directly run from the command line/terminal as follows:

Shell script - my_awesome_script.sh

#! /bin/sh
python nlp_is_awesome.py --everything=cool
 

sbatch --job-name=awesome --output=<your_dir>/awesome.out \
--error=<your_dir>/awesome.err --partition= studentbatch \
--time=1440 --signal=USR1@120 --nodes=1 --ntasks=1 --mem=50000 \
--cpus-per-task=4 --gpus=2 ./my_awesome_script.sh
*The job will be executed in a new shell, but from the same directory. This means that relative directories are sensitive to the location from which the job was launched.

(Go to top)

 

Using containers
Sometimes working with a conda env. is not enough and you require other tools to be installed. Slurm supports work with docker.

Running containers in batch mode:
1. Create slurm file like this (run.slurm):

​#!/bin/bash

#SBATCH --job-name=awesome
#SBATCH --output=sample.out
#SBATCH --error=sample.err
#SBATCH --time=150
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000 #50,000
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1

CMD=/path/to/script.sh
srun easy_ngc --cmd ${CMD} nvcr.io/nvidia/pytorch:20.12-py3 
Remember to change the partition, memory, time and everything to your parameters.

2. Create shell script to run (/path/to/script.sh). In the script you should insert every relevant thing - the command can't be more than 1 word! For example, ```script.sh -flag``` will NOT work!:

​​#!/bin/bash

source ~/.bashrc
​cd /path/to/python
​python awesome.py --param1 --param2 "lorem ipsum"
echo 'Done'
3. Verify that the script can be executed:

​> chmod ug+rx /path/to/script.sh
4. Run:

​> sbatch run.slurm
Possible error:
If you get an error that looks like this:

​usage: easy_ngc_impl [-h] [--cmd CMD] [--version VERSION] [--modules MODULES]

...

--ssh_key_prefix ...

...
It is most probably because you used a more than 1 word for the --cmd flag. There is no way to workaround that except for inserting everything into the script of step 2. So, get back to step 2, and then fix your command accordingly. For clarity, this lines:

​FULL_CMD=${SCRIPT_PATH}/${SCRIPT_NAME} \
--flag -f 'string with spaces'
srun easy_ngc --cmd ${FULL_CMD} nvcr.io/nvidia/pytorch
Will cause a problem. Change it like this - inside script.sh: 

/path/script.sh --flag parameter1 parameter2
And inside run.slurm:

​FULL_CMD=${SCRIPT_PATH}/script.sh
srun easy_ngc --cmd ${FULL_CMD} nvcr.io/nvidia/pytorch
 

DL containers from Nvidia (ngc):

> srun --gpus=2 --partition=studentrun --pty easy_ngc tensorflow --version=20.11-tf2-py3
The srun command works similarly to sbatch, but runs the job in interactive (blocking) mode. The --pty easy_ngc option tells slurm to run the job in a container, which emulates a shell in a safe environment (like Docker). Exiting the container (ctrl+d) ends the job and releases the resource. Besides the --pty option, srun and sbatch share almost all other options, such as --gpus.

srun documentation: https://slurm.schedmd.com/srun.html

 
> srun -G 3 --pty easy_ngc --cmd nvidia-smi nvcr.io/nvidia/pytorch
Use 3 GPUs and run the 'nvidia-smi' command using container with latest version of pytorch

> srun -G 2 --pty easy_ngc --jupyter mxnet
Use 2 GPUs and run jupyter notebook server using container with latest version of mxnet

> srun -G 5 --mem 120G --pty easy_ngc \
nvcr.io/nvidia/tensorflow:9.03-py3
Use 5 GPUs, allocate 120GB of system memory and use the March 2019 release of Tensorflow NGC container with python 3

> srun -G 2 --pty easy_ngc --modules=imagehash \
--packages=julia nvcr.io/nvidia/pytorch
Use 2 GPUs, run container with latest version of tensorflow, apt-get install julia and pip install imagehash inside container

> srun -G 2 --pty easy_ngc \
--modules=/home_dir/requirements.txt \
 nvcr.io/nvidia/tensorflow:20.11-tf2-py3
Use 2 GPUs, run container with latest version of tensorflow, pip install list of python packages with latest/specific version (You need to create your own requirements.txt file)

Non-DL containers (from Nvidia or elsewhere):
> srun -G 3 --pty easy_ngc \
nvcr.io/hpc/vmd:cuda9-ubuntu1604-egl-1.9.4a17
Pull the VMD HPC container from NGC and run a command line on it (with 3 GPUs)

>srun -G 5 --pty easy_ngc \
--cmd 'nvidia-smi' docker.io/library/ubuntu:19.10
Pull generic Ubuntu 19.10 container from Dockerhub and run nvidia-smi on it (with 5 GPUs)

(Go to top)

 

Using udocker
One might have a need to use a tool that is not installed on school servers. For example, R, gcc-11, python3.9 (without anaconda) and so on. For that, a tool named udocker is installed. Some examples of using udocker with slurm:

Interactive session of R

​udocker pull rocker/r-base
udocker create --name=r-container rocker/r-base
srun -p killable --pty udocker run r-container
 

R script - 2 options. You can also put these lines inside a slurm script:

srun -p killable udocker run \
--volume=/directory/of/R/script:/name/you/choose \
r-container R --vanilla -f /name/you/choose/script.r

​srun -p killable udocker run \
 --volume=/directory/of/R/script:/name/you/choose \
r-container Rscript /name/you/choose/script.r
 

Example of a slurm script that runs R container:

​#!/bin/bash

#SBATCH --job-name=awesome
#SBATCH --output=awesome.out
#SBATCH --partition=cpu-killable
srun udocker run --bindhome r-container Rscript ~/script.r
 

NOTE: when using Rscript, it is recommended to put the following line as first line in the script:

#!/usr/bin/Rscript
NOTE: You might need to change the permisiions of the script, like this:

​chmod ug+rx /path/to/script.r
(Go to top)

 

Migrating from condor
Here is an example of condor '.cmd' file and a matching '.slurm' script. An explanation about the conversion process appears after the example.

Condor .cmd file:

DIR = $ENV(HOME)/condor
Executable = $(DIR)/sample1
Log = sample1.log
Error = sample1.error.$(Process)
Output = sample1.output.$(Process)
notification = Always
Universe = Vanilla
Queue 4
 

A matching .slurm script:

#!/bin/bash

#SBATCH --job-name=sample1
#SBATCH --output=sample1.output.%A.%a # %j can be useful
#SBATCH --error=sample1.error.%A.%a
#SBATCH --partition=studentkillable
#SBATCH --mail-type=ALL,TIME_LIMIT_80 #notification
#SBATCH --time=1440 #minutes
#SBATCH --array=0-3 #not the same as --ntasks=4
#SBATCH --gres=gpu:1 

DIR=$HOME/condor
EXE=${DIR}/sample1
srun ${EXE}
Run the job with:

> sbatch <filename>.slurm
NOTE: sample1 execute bit has to be set:

> chmod ug+rx ~/condor/sample1
 

Useful environmanet variables are listed here:

https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES

Explanation of the conversion process:

The conversion process has 4 parts: 1. Direct conversion, 2. Addition of missing commands/instructions, 3. Elimination of unnecessary lines and 4. Re-ordering of the slurm script to its final form.

Note that in the slurm script the space were removed and the round brackets were replaced with curly brackets. Thst is because slurm scripts are also valid bash scripts.

Here is a table for the Direct conversion:

CMD file	slurm file
DIR = $ENV(HOME)/condor

Executable = $(DIR)/sample1

EXE="~/condor/sample1"
Error = <filename>.err.$(Process)	#SBATCH --error=<filename>.err.%a
Output = <filename>.out.$(Process)	#SBATCH --output=<filename>.out.%a
Arguments = $(Process)	ARGS=${SLURM_ARRAY_TASK_ID}
notification = Always	#SBATCH --mail-type=ALL
Queue 100	#SBATCH --array=0-99
 

 

 

 

 

 

 

Usually, one needs to add the following to the slurm script:

#!/bin/bash # as a first line 

#SBATCH --account=<relevant-account> 
#SBATCH --partition=<relevant-partition> 
#SBATCH --gres=gpu:n
#SBATCH -c <number of requested CPUs> # optional 

srun ${EXE} ${ARGS}
 

This lines has no direct replacement, and should be removed (except special cases):

Log = sample.log # can be replaced 
# using 
# scontrol show job <job number> 
# or
# sacct -j <job-number> 

Universe = Vanilla # should be removed
 

Finally, check that the slurm script is well formed:

1. The line starts with #! (Shebang) should be first line of the file.

2. After it should be all the lines that start with '#SBATCH'.

3. Then the variables.

4. And at last you should add the running line, the line starts with 'srun'.

The example we saw at the beginning show this exactly.

Possible errors

Usually not defining the right account/partition, running on the wrong cluster or not having permissions or resources.

(Go to top)

Common Errors/Problems
 Nodes stop recieving jobs. Check node status sinfo -R -l.  Nodes  are in "drain or drng" status: 
Nodes are being marked as drained by SLURM due to jobs that either crash or freeze as a result of Out of Memory (OOM) conditions. In these cases, the job often terminates without returning a proper exit code, which causes SLURM to interpret the node as unresponsive or in an error state, triggering the automatic drain.

Root Cause:

Jobs exceed the allocated memory limit and are killed by the kernel (OOM killer).

Since the job does not exit cleanly, SLURM cannot register a proper completion status.

SLURM then marks the node as drained to prevent further scheduling, assuming a hardware or system fault.

Recommendations to Avoid This:

Accurate Memory Requests: Ensure jobs request memory (--mem or --mem-per-cpu) that reflects actual usage. Underestimating memory leads to OOM kills.

Use --mem Limits with Caution: If memory usage is unpredictable, consider slightly overestimating to provide a buffer.

Enable Job Exit Codes: Where possible, ensure your job scripts or applications handle memory errors gracefully and exit with a proper code.

Monitor Usage: Use tools like sacct to analyze past memory usage and adjust future requests accordingly.

Node Health Checks: If a node is frequently drained, it may be worth checking for memory leaks or misbehaving jobs.