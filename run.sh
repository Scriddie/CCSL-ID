#!/usr/local_rwth/bin/bash

#SBATCH --job-name="sanity_exp" 
#SBATCH --output=output_%j.txt
#SBATCH --time=20:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

# TODO somehow sbatch run is not doing anything at all?

source env/bin/activate
python src/seq/exp_sanity.py