# 🌟 cross-modal-rag-implementation  
### A Multimodal Recipe Retrieval & Suggestion Engine (Text + Image RAG)

This project implements a **Cross-Modal Retrieval-Augmented Generation (RAG)** system for *food & recipe recommendations*.  
It retrieves supporting evidence from thousands of recipes using:

- **SBERT** for text embeddings  
- **CLIP** for image + text multimodal embeddings  
- **FAISS** for fast similarity search  
- **LLaMA 3.2 (3B-Instruct)** for reasoning and recipe generation  

The system supports **three types of queries**:
1. **Text-only RAG**  
2. **Image-only RAG**  
3. **Text + Image RAG**

---

## 🍽️ What This System Can Do

Given a text query like:

> "vegan pasta with broccoli"

or an uploaded image like:

> *(dish photo)*

or both together:

> "Something similar to this dish but spicier" + *(image)*

The system will:

1. Retrieve **top-K relevant recipe texts** (SBERT → FAISS)  
2. Retrieve **top-K relevant dish images** (CLIP → FAISS)  
3. Perform cross-modal retrieval  
   - text → image  
   - image → text  
   - image → image  
4. Build an intelligent **RAG prompt** containing multimodal evidence  
5. Use **LLaMA 3.2** to produce:
   - best-matching recipes  
   - ingredients lists  
   - explanations  
   - customization ideas  
   - diet-friendly alternatives  

---

## 🧠 System Architecture
```
  ┌──────────────────────┐
  │   User Query Input   │
  │  (Text / Image / Both)
  └───────────┬──────────┘
              │
              ▼
   ┌──────────────────────┐
   │  Cross-Modal Embedder│
   │ SBERT / CLIP Encoder │
   └───────────┬──────────┘
               │
               ▼
     ┌──────────────────┐
     │   FAISS Indexes  │
     │ text | image | clip_text
     └───────┬──────────┘
             │
             ▼
   ┌────────────────────────┐
   │  Retriever (4 modes)   │
   │ text→text              │
   │ text→image             │
   │ image→image            │
   │ image→text             │
   └──────────┬────────────┘
              │
              ▼
     ┌──────────────────┐
     │  Prompt Builder  │
     │  (Multimodal RAG)│
     └──────────┬───────┘
                │
                ▼
   ┌─────────────────────────┐
   │  LLaMA 3.2 (3B-Instruct) │
   │     Recipe Reasoning     │
   └───────────┬─────────────┘
               │
               ▼
     ┌────────────────────────┐
     │  Final Recipe Output   │
     └────────────────────────┘
```

---

## 📦 Project Structure
```
cross-modal-rag-implementation/
│
├── data/
│   ├── Food Images/
│   ├── Food Ingredients CSV
│
├── embeddings/
│   ├── sbert_text_embs.npy
│   ├── clip_text_embs.npy
│   ├── image_embs.npy
│   ├── ids.npy
│
├── indexes/
│   ├── text.index
│   ├── clip_text.index
│   ├── image.index
│
├── out/
│   ├── retrieved_evidence.txt
│   ├── generated_prompt.txt
│   ├── llm_output.txt
│
├── src/
│   ├── data_loader.py
│   ├── embedder.py
│   ├── build_embeddings.py
│   ├── build_index.py
│   ├── retriever.py
│   ├── prompt_builder.py
│   ├── llm_inference.py
│   ├── test_data_loading.py
│   ├── test_rag_prompt.py
│
└── README.md
```

---

## 🚀 Setup Instructions

### 0. Download Dataset

Download the dataset from Kaggle:

**[Food Ingredients and Recipe Dataset with Images](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)**

Extract the contents into the `data/` directory:
- Place images in `data/Food Images/`
- Place the CSV file in `data/`

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 📥 Step 1 — Load Dataset
```bash
python -m src.test_data_loading
```

This validates the recipe CSV + images and generates a `data_preview.txt`.

### 🧮 Step 2 — Build Embeddings
```bash
python -m src.build_embeddings
```

This generates:
- `sbert_text_embs.npy`
- `clip_text_embs.npy`
- `image_embs.npy`
- `ids.npy`

### ⚡ Step 3 — Build FAISS Indexes
```bash
python -m src.build_index
```

Creates:
- `indexes/text.index`
- `indexes/clip_text.index`
- `indexes/image.index`

### 🔍 Step 4 — Test Retrieval
```bash
python -m src.test_retrieval
```

Runs:
- text → text
- text → image
- image → image
- image → text

### 🤖 Step 5 — Full Multimodal RAG
```bash
python -m src.test_rag_prompt
```

Produces:
- `out/retrieved_evidence.txt`
- `out/generated_prompt.txt`
- `out/llm_output.txt`

---

## 🧠 LLM Used

This project uses:
```
meta-llama/Llama-3.2-3B-Instruct
```

To change the model, edit: `src/llm_inference.py`

---

## 🧪 Query Modes Supported

### ✅ 1. Text-Only Query

**Example:**
```
"vegan pasta with broccoli"
```

System runs:
- text→text
- text→image
- pseudo image→text
- pseudo image→image

### ✅ 2. Image-Only Query

**Example:**
```python
img = Image.open("my_dish.jpg")
```

System runs:
- image→text
- image→image

### ✅ 3. Combined Text + Image Query

**Example:**
```
"Something like this dish but spicier" + (uploaded image)
```

Perfect for personalization.

---

## 🎯 Future Improvements (Optional)

- Add BLIP / LLaVA image captions
- Cluster recipes by cuisine
- Add nutrition prediction
- Add FastAPI backend + React frontend
- Build a "chat with your pantry" system

---

## 📄 License

MIT License