#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=20
#SBATCH --job-name=percolation3d
#SBATCH --partition=clara

python 3d_percolation.py
