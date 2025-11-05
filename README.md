# Multimodal Search Engine (PE Core Edition)

This project implements a **text-to-image semantic search system** using **PE Core (Perception Embeddings by Meta)** as the embedding model and **Milvus Lite** as a vector database.  
The system indexes a dataset of images once and allows retrieving the most relevant images using natural language queries.

---

## Features

- **Text → Image Retrieval** using PE Core shared embedding space
- **Local vector database** powered by Milvus Lite
- **Cosine similarity search** for semantic relevance
- **Visualization** of result images
- **Modular design** supporting future model upgrades

---

## Installation

### 1. Clone and install PE Core
```bash
git clone https://github.com/facebookresearch/perception_models.git
cd perception_models
pip install -e .
```

### 2. Install project dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Create / Index the Vector Database
Edit the image dataset path in `create_db.py`, then run:

```bash
python create_db.py
```

This will:
- Encode images using PE Core
- Store embeddings + file paths inside Milvus Lite

### Step 2 — Search the Database
```bash
python search_db.py
```

You will be prompted to enter a text query:

```
Enter search query: a red motorcycle on a street
```

Matching images will be displayed.

---

## Example Query

**Input**
```
dog running in a field
```

**Output**
- Retrieves and displays the most semantically similar images
- Shows similarity scores

---

## Project Structure

```
.
├── create_db.py             # Embeds images + creates vector DB
├── search_db.py             # Performs text→image semantic retrieval
├── requirements.txt         # Dependencies
├── notebook.ipynb           # Optional exploration notebook
└── README.md                # Documentation
```

---

## How It Works

| Step | Description |
|------|-------------|
| 1 | PE Core encodes images into embeddings |
| 2 | PE Core encodes text queries into the same embedding space |
| 3 | Embeddings are normalized for cosine similarity |
| 4 | Milvus Lite performs nearest-neighbor lookup |
| 5 | Results are displayed with scores |

---

## Future Enhancements

- Support for additional PE Core variants (ViT-L, ConvNext)
- Optional Streamlit or Gradio web UI
- Ability to scale to full Milvus Server for large datasets
- Potential extension to multimodal retrieval (video/audio)
