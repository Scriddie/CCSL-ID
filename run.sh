#!/usr/local_rwth/bin/bash

#SBATCH --job-name="sanity_exp" 
#SBATCH --output=output_%j.txt
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8

# TODO
# SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=4G

# TODO do more cpus make it slower or do they just take longer to set up?
# TODO more nodes or more tasks per node??
# TODO paralellism doesn't seem to work :/

source env/bin/activate
python src/seq/exp_sanity.py