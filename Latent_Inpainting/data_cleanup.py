from PIL import Image
import os
from tqdm import tqdm

data_path = "/scratch/bmutembei36/Data/Data/training/"
files = [f for f in os.listdir(data_path) if f.endswith('.png')]

for f in tqdm(files):
    path = os.path.join(data_path, f)
    try:
        with Image.open(path) as img:
            img.verify() 
    except Exception:
        print(f"Deleting corrupted file: {path}")
        os.remove(path)