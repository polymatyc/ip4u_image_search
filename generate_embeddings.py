import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm
import pickle
import torch

# Paths
METADATA_PATH = "../database/metadata.csv"
EMBEDDINGS_PATH = "../database/embeddings.npy"
PATHS_PATH = "../database/image_paths.pkl"

# Load model (GPU automatically if available)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('clip-ViT-B-32', device=device)
print("Using device:", device)
# Load metadata
df = pd.read_csv(METADATA_PATH)

image_paths = df['image_path'].tolist()

embeddings = []
valid_paths = []

print("Generating embeddings...")

for path in tqdm(image_paths):
    try:
        image = Image.open(path).convert("RGB")
        embedding = model.encode(image, convert_to_numpy=True)
        embeddings.append(embedding)
        valid_paths.append(path)
    except:
        continue

embeddings = np.array(embeddings)

# Save embeddings
np.save(EMBEDDINGS_PATH, embeddings)

# Save corresponding image paths
with open(PATHS_PATH, "wb") as f:
    pickle.dump(valid_paths, f)

print("Done. Total embeddings:", len(embeddings))