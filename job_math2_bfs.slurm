#!/bin/bash
#SBATCH --job-name=Math_ToT 
#SBATCH --nodes=1  
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=1      
##SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1   
#SBATCH --time=09:00:00  
#SBATCH --mem-per-cpu=10G         # memory per cpu-core (4G is default)
##SBATCH --array=0-8              # job array with index values 0, 1, 2,...
#SBATCH -p gpu-ee
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=jx0800@princeton.edu
#SBATCH --output=slurm_logs/%A_%a

module purge
module load anaconda3/2024.6
conda activate tot
sh scripts/MATH2/bfs.sh