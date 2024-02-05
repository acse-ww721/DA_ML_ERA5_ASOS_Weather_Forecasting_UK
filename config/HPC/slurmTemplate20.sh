ww721@ese-hivemind:/raid/hivemind$ cat slurmTemplate20.sh 
#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
# e.g. The jobname below is "example"
#SBATCH --job-name=example

# Each job will utilise all of brainstorm's resources.
# Here, we ask for 1 node with exlusive
# brainstorm has 256 CPU cores and 8 A100 GPUs.

####--cpus-per-task 8 --gres=gpu:1 --mem-per-cpu 1500 --time 24:00:00 --pty bash -i
#SBATCH -e slurm-%j.err              # File to redirect stderr
#SBATCH -o slurm-%j.out              # File to redirect stdout
#SBATCH --mem-per-cpu=1500           # Memory per processor
#SBATCH --time=0-00:04:00            # The walltime
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=2                   # Number of tasks
#SBATCH --ntasks-per-socket=1        # Maximum number of tasks on each socket

#SBATCH --cpus-per-task=5            # Number of CPU cores per task
#SBATCH --gres=gpu:2                 # Number of GPUs to allocate to this job

# This is where the actual work is done.

#nvidia-smi
#nvcc --version
source ~/miniconda3/etc/profile.d/conda.sh
conda activate testgpu
#numba -s

# These are individual tasks
python xavier.py 0&
python xavier.py 1&
wait

# Finish the script
exit 0
