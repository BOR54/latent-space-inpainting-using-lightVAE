#!/bin/sh
#
# [scan_path.sbatch]
#
#SBATCH --job-name=testing_lightVAE              # Job name
#SBATCH --output=testing_VAE.out            # Output file name
#SBATCH --error=testing_VAE_stderr.txt
#SBATCH --chdir=/home/bmutembei36/Latent_Inpainting
#SBATCH --export=ALL                  # Export all environment variables
#SBATCH --time=120:01:00               # Set max runtime of job = 1 minute
#SBATCH --mem=10G                      # Request 4 gigabytes of memory
#SBATCH --mail-type=FAIL    # Send email notifications
#SBATCH --mail-user=bmutembei@ou.edu  # email TO
#SBATCH --partition=octiv              # Specify the partition to submit the job to



srun python /home/bmutembei36/Latent_Inpainting/open-litevae/inference.py \
  --config /home/bmutembei36/Latent_Inpainting/open-litevae/configs/olitevaeB_im_f8c12.yaml \
  --ckpt_dir /scratch/bmutembei36/litevae_runs/checkpoints/ \
  --ckpt /scratch/bmutembei36/litevae_runs/checkpoints/last.ckpt \
  --data_root /scratch/bmutembei36/Data/Data \
  --split testing \
  --out_dir /scratch/bmutembei36/litevae_infer_outputs \
  --batch_size 8 \
  