#!/bin/sh
#
# [scan_path.sbatch]
#
#SBATCH --job-name=train_lightVAE              # Job name
#SBATCH --output=train_lightVAE.out            # Output file name
#SBATCH --error=train_VAE_stderr.txt
#SBATCH --chdir=/home/bmutembei36/Latent_Inpainting
#SBATCH --export=ALL                  # Export all environment variables
#SBATCH --time=120:01:00               # Set max runtime of job = 1 minute
#SBATCH --mem=20G                      # Request 4 gigabytes of memory
#SBATCH --mail-type=FAIL    # Send email notifications
#SBATCH --mail-user=bmutembei@ou.edu  # email TO
#SBATCH --partition=octiv              # Specify the partition to submit the job to



srun python /home/bmutembei36/Latent_Inpainting/open-litevae/train_lightvae.py \
  --config /home/bmutembei36/Latent_Inpainting/open-litevae/configs/olitevaeB_im_f8c12.yaml \
  --data_root /scratch/bmutembei36/Data/Data \
  --out_dir /scratch/bmutembei36/litevae_runs \
  --batch_size 8 --num_workers 4 \
  --max_epochs 50 --precision bf16-mixed
