#!/usr/local_rwth/bin/bash

#SBATCH --job-name="sanity_exp" 
#SBATCH --output=output_%j.txt
#SBATCH --time=20:00 
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G

source env/bin/activate
python src/seq/exp_sanity.py