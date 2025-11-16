# Cross-Modal RAG — Multimodal Misinformation Detection  
**Retrieve + Reason over Text & Images using CLIP, SBERT, FAISS & LLMs**

This project implements a **Cross-Modal Retrieval-Augmented Generation (RAG)** system that takes a **claim + image pair**, retrieves the most relevant textual and visual evidence, builds an evidence-aware prompt, and sends it to an LLM to determine whether the claim is:
- **True**
- **False**
- **Uncertain**

---

## 🔍 Core Idea

We don't rely on the claim text alone.

We:
1. Embed **all text** with SBERT  
2. Embed **all images** with CLIP  
3. Build two FAISS indexes  
4. Retrieve:
   - Top-K similar texts  
   - Top-K similar images  
   - Cross-modal image matches  
5. Build a fused RAG prompt  
6. Send to an LLM for reasoning  

---

## 🧠 Architecture

```
(Claim + Image)
│
▼
[Cross-Modal Retriever]
│         │         │
Text→Text  Img→Img  Text→Img
▼         ▼         ▼
Top-K Evidence Samples
│
▼
[Prompt Builder]
│
▼
LLM
│
▼
True / False / Uncertain
```

---

## 📂 Project Structure

```
cross-modal-rag/
│
├── data/
│   ├── images/
│   └── cache_train.csv
│
├── embeddings/
│   ├── ids.npy
│   ├── text_embs.npy
│   └── image_embs.npy
│
├── indexes/
│   ├── text.index
│   └── image.index
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
│   ├── test_retrieval.py
│   └── test_rag_prompt.py
│
└── requirements.txt
```

---

## 🛠 Installation

```bash
git clone <repo>
cd cross-modal-rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚦 Step 1 — Load Dataset & Download Images

```bash
python -m src.test_data_loading
```

This:
- downloads image+text pairs
- saves them to `data/images/`
- caches metadata in `data/cache_train.csv`

---

## 🔡 Step 2 — Generate Embeddings

```bash
python -m src.build_embeddings
```

Produces:
```
embeddings/ids.npy
embeddings/text_embs.npy
embeddings/image_embs.npy
```

---

## 📦 Step 3 — Build FAISS Indexes

```bash
python -m src.build_index
```

Outputs:
```
indexes/text.index
indexes/image.index
```

---

## 🔍 Step 4 — Test Retrieval

```bash
python -m src.test_retrieval
```

Validates:
- Text → Text search
- Image → Image similarity
- Text → Image via CLIP

---

## 🤖 Step 5 — Run Full RAG

```bash
python -m src.test_rag_prompt
```

This will:
1. Retrieve multimodal evidence
2. Construct a fact-checking prompt
3. Send to LLM
4. Return a verdict + explanation

Example:
```
Verdict: False  
Explanation: Retrieved evidence contradicts the claim.
```

---

## 🔌 LLM Model Switching

Edit here:
```bash
src/llm_inference.py
```

Change:
```python
model_name = "microsoft/phi-2"
```

to:
```python
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# or
model_name = "microsoft/Phi-3-mini-4k-instruct"
# or any HF model
```

---

## 🧩 Integrating Your Fine-Tuned Model

Once your custom LLM is trained:

1. Push it to HuggingFace OR
2. Load it locally:

```python
model = AutoModelForCausalLM.from_pretrained("./my_model")
```

Everything else stays the same.

---

## 📌 Requirements

- Python 3.9+
- PyTorch (+ CUDA)
- SentenceTransformers
- FAISS
- CLIP
- PIL
- Transformers
- Accelerate

---

## ➕ Extensions

You can extend this to:
- Larger datasets (9k → 100k+)
- Add image captions (BLIP / LLaVA)
- Use GPT-4V or LLaVA-Next
- Run via FastAPI / Streamlit / Gradio

---

## 📄 License

MIT License

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.
