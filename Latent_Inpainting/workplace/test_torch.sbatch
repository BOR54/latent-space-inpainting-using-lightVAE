#!/bin/bash
#
#SBATCH --partition=octiv
#SBATCH --output=test_gpu_stout.txt
#SBATCH --error=test_gpu_stderr.txt
#SBATCH --chdir=/home/bmutembei36/Latent_Inpainting
#SBATCH --ntasks=3
#SBATCH --mem=1G
#SBATCH --time=00:10:00

# module load Python/3.10.8-GCCcore-12.2.0
#python "/home/bmutembei36/Latent_Inpainting/workplace/test_torch.py"
python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.version.cuda, 'built=', torch.backends.cuda.is_built(), 'avail=', torch.cuda.is_available())"

python "/home/bmutembei36/Latent_Inpainting/workplace/test_gpu.py"