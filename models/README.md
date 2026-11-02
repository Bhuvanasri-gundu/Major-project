# Model Registry & Storage Documentation

This directory (`models/`) serves as the dedicated, canonical repository for **all locally saved model weights, checkpoints, trained/fine-tuned models, and serialized model packages** in the project:

**Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation in mHealth Research**

---

## 1. Model Inventory

| Model Directory / Package | Model Class / Architecture | Serialization Format | Target Purpose | Size on Disk | Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`bertopic_model/`](bertopic_model/) | **BERTopic (c-TF-IDF + UMAP + HDBSCAN)** | `safetensors` + `json` | Unsupervised biomedical topic discovery across 12,642 screened mHealth publications (63 topics + outlier) | ~3.7 MB | Trained in Phase 1 (`src/topic_modeling.py` / `notebooks/05_bertopic_topic_discovery.ipynb`) |

---

## 2. Directory Structure & File Breakdown: `models/bertopic_model/`

```text
models/bertopic_model/
├── config.json                  # BERTopic configuration & pipeline parameters (316 bytes)
├── ctfidf.safetensors           # Class-based TF-IDF sparse matrix weights (3,053,996 bytes)
├── ctfidf_config.json           # CountVectorizer vocabulary & c-TF-IDF hyperparameters (288,094 bytes)
├── topics.json                  # Document-to-topic cluster assignment mappings (186,017 bytes)
└── topic_embeddings.safetensors # 768-dimensional topic centroid embeddings in SPECTER2 space (196,696 bytes)
```

### Component Details
1. **`config.json`**:
   - `version`: BERTopic package serialization version.
   - `calculate_probabilities`: False (deterministic cluster assignment).
   - `nr_topics`: 63 active topics + topic `-1` (unassigned/outlier corpus).
2. **`ctfidf.safetensors`**:
   - Safe, zero-copy tensor representation of word importance scores per discovered topic cluster.
   - Built with `serialization="safetensors"` to prevent arbitrary code execution vulnerabilities associated with legacy `.pkl`/`.joblib` formats.
3. **`ctfidf_config.json`**:
   - Tokenizer parameters: `ngram_range = (1, 2)`, English stopwords excluded, minimum document frequency threshold $df \ge 10$.
4. **`topics.json`**:
   - Topic representation labels and ranked c-TF-IDF keywords (e.g., Topic 0: *diabetes, glucose, insulin*; Topic 1: *depression, mental, anxiety*; Topic 2: *wearable, sensors, physical activity*).
5. **`topic_embeddings.safetensors`**:
   - Dense centroid representations in 768-dimensional SPECTER2 space, enabling direct semantic distance computation against MeSH concepts.

---

## 3. External Pretrained Models & API Integration (Non-Local)

Following strict research reproducibility guidelines, external transformer foundation models and cloud APIs are accessed on-demand with local caching in the standard user cache, rather than duplicating multi-gigabyte external weights inside the repository:

| Model Name / Identifier | Framework | Role in Pipeline | Hosting / Provenance |
| :--- | :--- | :--- | :--- |
| `allenai/specter2_base` | Hugging Face Transformers | Scientific document representation (768-dim) in Phase 1 | Downloaded via Hugging Face Hub (cached) |
| `sentence-transformers/all-MiniLM-L6-v2` | Sentence-Transformers | Abstract chunk dense embeddings (384-dim) for FAISS retrieval & deduplication | Downloaded via Hugging Face Hub (cached) |
| `cross-encoder/nli-deberta-v3-small` | Cross-Encoder / DeBERTa | Natural Language Inference premise-hypothesis validation (Entailment / Contradiction) | Downloaded via Hugging Face Hub (cached) |
| `gemini-3.1-flash-lite` / `gemini-2.0-flash` | Google GenAI SDK | Structured hypothesis synthesis & LLM Judge rubric evaluation | Cloud API endpoint via `GEMINI_API_KEY` |

---

## 4. Architectural Separation: Models vs. Artifacts

To prevent misclassification:
- **`models/` (This Directory)**: Houses strictly model weights, parameters, and serialized model state packages (such as `bertopic_model/`).
- **`outputs/`**: Contains pipeline results and downstream representations:
  - `outputs/faiss_index/`: Dense FAISS vector index binary (`faiss_chunk_index.bin`) and chunk metadata (`faiss_chunk_metadata.json`).
  - `outputs/*.npy`: Raw static embedding matrices (`document_embeddings_specter2.npy`, `concept_embeddings_specter2.npy`).
  - `outputs/*.csv` / `outputs/*.json`: Candidate research gaps, preprocessed corpora, generated hypotheses, and validation outputs.
- **`data/`**: Houses raw and screened PubMed CSV files.

---

## 5. Loading Instructions & Code Usage

### Loading `bertopic_model` in Python
```python
from pathlib import Path
from bertopic import BERTopic

# Path to the serialized model
model_path = Path("models/bertopic_model")

# Load model using safetensors backend
topic_model = BERTopic.load(str(model_path))

# Inspect topic metadata
topic_info = topic_model.get_topic_info()
print(f"Loaded {len(topic_info)} topics.")
print(topic_info.head(5)[["Topic", "Count", "Name"]])

# Query keywords for a specific topic (e.g. Topic 0)
keywords = topic_model.get_topic(0)
for word, score in keywords[:5]:
    print(f"  {word}: {score:.4f}")
```

### Loading via Project Helper
```python
from src.topic_modeling import load_saved_bertopic_model

topic_model = load_saved_bertopic_model("models/bertopic_model")
```

---

## 6. Software Dependencies

Loading and executing models in this directory requires the following packages (specified in `requirements.txt`):
- `bertopic>=0.16.0`
- `safetensors>=0.4.0`
- `scikit-learn>=1.3.0`
- `umap-learn>=0.5.4`
- `hdbscan>=0.8.33`
- `torch>=2.1.0`
