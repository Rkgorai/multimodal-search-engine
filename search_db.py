import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pymilvus import connections, Collection

# -------------------------------------------------
# 1. Import perception model modules
# -------------------------------------------------
sys.path.append('./perception_models')
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms_pe

# -------------------------------------------------
# 2. CONFIG — must match create.py
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "PE-Core-S16-384"  # must match create.py
DB_PATH = os.path.join('/kaggle/working/', model_name, "frame_source_db", "milvus.db")
COLLECTION_NAME = "frame_embeddings_db"

IMAGE_FOLDER = "/kaggle/input/coco-2017-dataset/coco2017/val2017"  # image source folder
TOP_K = 5  # Number of results to show

# -------------------------------------------------
# 3. Load CLIP Model
# -------------------------------------------------
print(f"[+] Loading CLIP model: {model_name}")
model = pe.CLIP.from_config(model_name, pretrained=True)
model = model.eval().to(device)
tokenizer = transforms_pe.get_text_tokenizer(model.context_length)

# -------------------------------------------------
# 4. Connect to Milvus Lite
# -------------------------------------------------
if connections.has_connection("default"):
    connections.disconnect("default")

print(f"[+] Connecting to Milvus Lite DB: {DB_PATH}")
connections.connect(alias="default", uri=DB_PATH)
collection = Collection(name=COLLECTION_NAME, using="default")
collection.load()

print(f"[✓] Connected to Milvus Lite DB: {DB_PATH}")
print(f"[✓] Collection loaded: {COLLECTION_NAME}")

# -------------------------------------------------
# 5. Encode Text Query
# -------------------------------------------------
def encode_text(query: str):
    text_tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.squeeze().cpu().tolist()

# -------------------------------------------------
# 6. Search & Plot Results
# -------------------------------------------------
def search_and_show(query, top_k=TOP_K):
    query_vector = encode_text(query)
    search_params = {"metric_type": "COSINE", "params": {}}

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["id"],
    )

    if not results or not results[0]:
        print("[!] No matches found.")
        return []

    hits = results[0]
    filepaths, scores = [], []

    for hit in hits:
        img_id = hit.id
        possible_paths = [
            os.path.join(IMAGE_FOLDER, f"{img_id}.jpg"),
            os.path.join(IMAGE_FOLDER, f"{img_id}.jpeg"),
            os.path.join(IMAGE_FOLDER, f"{img_id}.png"),
        ]
        found_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if found_path:
            filepaths.append(found_path)
            scores.append(hit.score)
        else:
            print(f"[!] Missing file for ID: {img_id}")

    # --- Plot results ---
    if not filepaths:
        print("[!] No valid image files to display.")
        return []

    plt.figure(figsize=(15, 3))
    for i, (fp, sc) in enumerate(zip(filepaths, scores)):
        try:
            img = Image.open(fp).convert("RGB")
            plt.subplot(1, len(filepaths), i + 1)
            plt.imshow(img)
            plt.title(f"Score: {sc:.3f}", fontsize=9)
            plt.axis("off")
        except Exception as e:
            print(f"[!] Error displaying {fp}: {e}")

    plt.suptitle(f"Top {len(filepaths)} matches for: '{query}'", fontsize=13)
    plt.tight_layout()
    plt.show()

    return filepaths

# -------------------------------------------------
# 7. Main Loop
# -------------------------------------------------
if __name__ == "__main__":
    try:
        while True:
            query = input("\nEnter your search query (or 'exit' to quit): ").strip()
            if query.lower() == "exit":
                break

            print(f"\n[INFO] Searching for: '{query}' ...")
            top_images = search_and_show(query, TOP_K)

            if top_images:
                print("\n[✓] Top Retrieved Image Paths:")
                for path in top_images:
                    print("  ", path)

    except Exception as e:
        print(f"[✗] Error: {e}")
    finally:
        if connections.has_connection("default"):
            connections.disconnect("default")
        print("[✓] Disconnected from Milvus.")
