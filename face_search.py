import faiss
import torch
import clip
import os
import pandas as pd

# Read file KNN index 
df = pd.read_parquet(".\data\embedding_folder\metadata\metadata_0.parquet")
image_list = df["image_path"].tolist()
ind = faiss.read_index(".\data\knn.index")

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# Search image
def search_face(image):
    image_tensor = preprocess(image)
    image_features = model.encode_image(torch.unsqueeze(image_tensor.to(device), dim=0))
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_embeddings = image_features.cpu().detach().numpy().astype('float32')
    D, I = ind.search(image_embeddings, 1)
    if D[0][0] > 0.8: 
        name = os.path.basename(os.path.dirname(image_list[I[0][0]])) 
        print("Name:",os.path.basename(os.path.dirname(image_list[I[0][0]])))
        print("Similarity:",D[0][0])
        return name
