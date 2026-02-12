#!/bin/bash
#SBATCH --job-name=lightVAE_gan
#SBATCH --output=train_lightVAE_gan.out
#SBATCH --error=train_lightVAE_gan.err
#SBATCH --chdir=/home/bmutembei36/Latent_Inpainting
#SBATCH --export=ALL
#SBATCH --time=120:01:00
#SBATCH --mem=20G
#SBATCH --partition=octiv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bmutembei@ou.edu

srun python /home/bmutembei36/Latent_Inpainting/open-litevae/training_gan.py \
  --config /home/bmutembei36/Latent_Inpainting/open-litevae/configs/olitevaeB_im_f8c12_gan.yaml \
  --data_root /scratch/bmutembei36/Data/Data \
  --out_dir /scratch/bmutembei36/litevae_runs \
  --batch_size 8 --num_workers 8 \
  --max_epochs 50 --precision bf16-mixed \
  --devices 1