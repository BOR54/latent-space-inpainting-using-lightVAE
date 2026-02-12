#!/bin/sh
#
# [scan_path.sbatch]
#
#SBATCH --job-name=cleaning              # Job name
#SBATCH --output=cleaning.out            # Output file name
#SBATCH --error=cleaning.txt
#SBATCH --chdir=/home/bmutembei36/Latent_Inpainting
#SBATCH --export=ALL                  # Export all environment variables
#SBATCH --time=120:01:00               # Set max runtime of job = 1 minute
#SBATCH --mem=10G                      # Request 4 gigabytes of memory
#SBATCH --mail-type=FAIL    # Send email notifications
#SBATCH --mail-user=bmutembei@ou.edu  # email TO
#SBATCH --partition=octiv              # Specify the partition to submit the job to



python "/home/bmutembei36/Latent_Inpainting/data_cleanup.py"
  