#!/bin/bash
#
#
# ==== SLURM part (resource manager part) ===== #
#
## Metadata configuration
#
#SBATCH --job-name=RLBreakout             # Job name
#SBATCH --mail-type=ALL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=acarraro@sissa.it     # Where to send mail
#
## CPU resources configuration
#
#SBATCH --ntasks=1                  # Number of MPI ranks (1 for MPI serial job)
#SBATCH --cpus-per-task=40            # Number of threads per MPI rank (2xcores on the new nodes, 1xcores on the old ones) 
#SBATCH --nodes=1                    # Number of nodes
#[optional] #SBATCH --ntasks-per-node=1          # How many tasks on each node
#[optional] #SBATCH --ntasks-per-socket=1        # How many tasks on each CPU or socket
#[optional] #SBATCH --distribution=cyclic:cyclic # Distribute tasks cyclically on nodes and sockets
#
## Other resources configuration (e.g. GPU)
#
#[not configured yet] #SBATCH --gpus:2                     # GPUs per job
#
## Memory configuration
#
#SBATCH --mem=0                # Memory per node (63500 on the new ones, 40000 on the old ones); incompatible with --mem-per-cpu
#[optional] #SBATCH --mem-per-cpu=4000mb         # Memory per core; incompatible with --mem
#
## Queue, Walltime and Output
#
#[non mi serve] #SBATCH --array=001-010:001%6
#SBATCH -p regular2,regular1,gpu1,gpu2  # Partition (queue) to be used
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec
#SBATCH --output=%x.o%j                 # Standard output log in TORQUE-style -- WARNING: %x requires a new enough SLURM. Use %j for regular jobs and %A-%a for array jobs
#SBATCH --error=%x.e%j                  # Standard error  log in TORQUE-style -- WARNING: %x requires a new enough SLURM. Use %j for regular jobs and %A-%a for array jobs
#
# ==== End of SLURM part (resource manager part) ===== #
#
#
# ==== Modules part (load things) =====
#

module purge
module load python3/3.6

#
# ==== End of Modules part (load things) =====
#
#

# Move to the working directory
cd $SLURM_SUBMIT_DIR

# ==== JOB COMMANDS =====

source venv/bin/activate

python3 Breakout_script.py

# ==== END OF JOB COMMANDS =====


# Wait for processes, if any.
echo "Waiting for all the processes to finish..."
wait
