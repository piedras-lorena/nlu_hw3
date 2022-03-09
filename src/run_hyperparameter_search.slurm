#!/bin/bash
#SBATCH --job-name=roberta_hyperparameter_search
#SBATCH --open-mode=append
#SBATCH --output=/scratch/lp2535/spring2022/nlu/hw/nlu_hw3/out/%j_%x.out
#SBATCH --error=/scratch/lp2535/spring2022/nlu/hw/nlu_hw3/err/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.
singularity exec --bind /scratch/lp2535 --nv --overlay /scratch/lp2535/overlay-25GB-500K.ext3:ro /scratch/lp2535/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/lp2535/spring2022/nlu/hw/nlu_hw3/src/run_hyperparameter_search.py /scratch/lp2535/spring2022/nlu/hw/nlu_hw3/data/BoolQ
"
