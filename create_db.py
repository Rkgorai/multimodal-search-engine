import os
import sys
import torch
from tqdm import tqdm
from PIL import Image
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Add model paths
sys.path.append('./perception_models')
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms_pe

# -----------------------------
# CONFIG
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "PE-Core-S16-384"  # Must match search.py
DB_ROOT = "/workspace/rahul/final/milvusdb"
IMAGE_FOLDER = "/kaggle/input/coco-2017-dataset/coco2017/val2017"  # source of frames
COLLECTION_NAME = "frame_embeddings_db"

# Path where Milvus Lite DB will be saved
DB_PATH = os.path.join('/kaggle/working/', model_name, "frame_source_db", "milvus.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# -----------------------------
# 1. Load CLIP Model
# -----------------------------
print("[+] Loading CLIP model:", model_name)
model = pe.CLIP.from_config(model_name, pretrained=True)
model = model.eval().to(device)

preprocess = transforms_pe.get_image_transform(model.image_size)

# -----------------------------
# 2. Connect to Milvus Lite
# -----------------------------
print(f"[+] Connecting to Milvus Lite DB: {DB_PATH}")
# Disconnect if same alias is already active
if connections.has_connection("default"):
    connections.disconnect("default")

connections.connect(alias="default", uri=DB_PATH)


# -----------------------------
# 3. Define Collection Schema
# -----------------------------
if utility.has_collection(COLLECTION_NAME):
    print("[✓] Existing collection found. Loading...")
    collection = Collection(COLLECTION_NAME)
else:
    print("[+] Creating new collection schema...")

    # Get embedding dimension dynamically
    dummy_img = next(f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    dummy_path = os.path.join(IMAGE_FOLDER, dummy_img)
    dummy_tensor = preprocess(Image.open(dummy_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        dummy_emb = model.encode_image(dummy_tensor)
        emb_dim = dummy_emb.shape[-1]

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=emb_dim),
    ]
    schema = CollectionSchema(fields, description="Frame-level CLIP embeddings")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print("[✓] New collection created:", COLLECTION_NAME)

# -----------------------------
# 4. Index Images
# -----------------------------
def encode_image(image_path):
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().tolist()

def index_images(image_dir, collection):
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    print(f"[+] Found {len(image_files)} images to index.")

    ids, vectors = [], []
    for img_file in tqdm(image_files, desc="Embedding images"):
        img_path = os.path.join(image_dir, img_file)
        try:
            emb = encode_image(img_path)
            img_id = os.path.splitext(img_file)[0]  # filename without extension
            ids.append(img_id)
            vectors.append(emb)
        except Exception as e:
            print(f"[!] Skipping {img_file}: {e}")

    if len(ids) > 0:
        collection.insert([ids, vectors])
        print(f"[✓] Inserted {len(ids)} image embeddings into Milvus collection.")
    else:
        print("[!] No valid images were indexed.")

# -----------------------------
# 5. Build Index
# -----------------------------
def create_index(collection):
    print("[+] Creating index...")
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print("[✓] Index created successfully.")

# -----------------------------
# 6. Main Logic
# -----------------------------
if __name__ == "__main__":
    try:
        # Only index if empty
        if collection.num_entities == 0:
            index_images(IMAGE_FOLDER, collection)
            create_index(collection)
        else:
            print(f"[✓] Collection already has {collection.num_entities} entries. Skipping indexing.")

        collection.flush()
        print("[✓] All embeddings flushed and ready for search.")

    except Exception as e:
        print(f"[✗] Error during indexing: {e}")

    finally:
        if connections.has_connection("default"):
            connections.disconnect("default")
            print("[✓] Disconnected from Milvus.")
