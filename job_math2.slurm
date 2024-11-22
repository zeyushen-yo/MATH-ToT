#!/bin/bash
#SBATCH --job-name=Math_ToT 
#SBATCH --nodes=1  
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=1      
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=mig
#SBATCH --gres=gpu:1   
#SBATCH --time=24:00:00  
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=zs7353@princeton.edu
module purge
module load anaconda3/2024.2
conda activate Math_ToT
sh scripts/MATH2/bfs.sh