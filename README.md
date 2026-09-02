# Mitigating Hallucinations in Generative AI
### The Effectiveness of Retrieval-Augmented Generation and Fine-Tuning

> BSc Thesis — Giray Kurt
> 
> Department of Computer Science, Vrije Universiteit Amsterdam — 2025

[![Research Paper](https://img.shields.io/badge/Research_Paper-PDF-red)](./BSc_Thesis__Giray_Kurt.pdf)
[![Fine-Tuned Model](https://img.shields.io/badge/HuggingFace-Mist--FT--2106--4bit-yellow)](https://huggingface.co/girayzkrt/Mist-FT-2106-4bit)

---

## Overview

This research investigates hallucination mitigation in Large Language Models (LLMs) applied to the biomedical domain. The project benchmarks four distinct generation strategies on a medical question-answering task using the PubMedQA dataset, comparing how **Retrieval-Augmented Generation (RAG)** and **fine-tuning** independently and combined affect factual accuracy relative to a vanilla LLM baseline.

The base model throughout is **Mistral-7B-Instruct**, deployed locally via [Ollama](https://ollama.com) for vanilla and RAG experiments, and as a 4-bit quantized fine-tuned checkpoint (`girayzkrt/Mist-FT-2106-4bit`) for fine-tuning experiments.

---

## Research Design

Four conditions are evaluated:

| Condition | Description |
|-----------|-------------|
| **Vanilla LLM** | Mistral-7B-Instruct with no external context |
| **RAG** | Mistral-7B-Instruct augmented with retrieved PubMed abstracts |
| **Fine-Tuned (FT)** | Mistral-7B fine-tuned on biomedical literature (`Mist-FT-2106-4bit`) |
| **FT + RAG** | Fine-tuned model augmented with retrieved PubMed context |

Answers are evaluated against PubMedQA ground-truth labels using three complementary metrics: **BERTScore**, **Semantic Similarity**, and **ROUGE-L**.

---

## Repository Structure

```
Hallu-Miti-Gen-AI/
├── scripts/
│   ├── parser.py                        # PMC XML → JSONL parser
│   ├── split_data.py                    # Dataset split utility
│   └── preprocessing/
│       ├── download_data.py             # Download PMC bulk XML archives
│       ├── extract_data.py              # Extract tar.gz archives
│       ├── 0_remove_blank_title_abstract.py
│       └── 2_text_cleaning.py
│
├── src/
│   ├── embedding/
│   │   └── pipeline.py                  # Chunking + embedding + Qdrant ingestion
│   │
│   ├── generation/
│   │   ├── vanilla/vanilla_llm.py       # Condition 1: Vanilla LLM
│   │   ├── rag/rag_llm.py               # Condition 2: RAG
│   │   ├── fine-tuned/fine_tuned_llm.py # Condition 3: Fine-tuned
│   │   └── fine_rag/fine_and_rag.py     # Condition 4: Fine-tuned + RAG
│   │
│   ├── evaluation/
│   │   ├── bertscore/
│   │   │   ├── get_bert.py              # BERTScore computation
│   │   │   └── bert_comparison.py       # Cross-condition comparison
│   │   └── semantic_similarity_and_rouge_l/
│   │       └── semantic_scoring.py      # Semantic similarity + ROUGE-L + plots
│   │
│   └── pca/
│       ├── dimension_analysis.ipynb
│       └── reduce_dimensions.py
│
├── notebooks/
│   └── embedding.ipynb                  # Embedding exploration
│
├── playground/                          # Experiments (FAISS, MiniLM, Qwen)
│   ├── embed_faiss.ipynb
│   ├── miniLM.py
│   ├── qwen1-5-1-8b-chat.ipynb
│   ├── retrieve.py
│   └── tokens.ipynb
│
└── BSc_Thesis__Giray_Kurt.pdf
```

---

## Pipeline

### 1. Data Collection

PubMed Central (PMC) open-access non-commercial articles are downloaded in bulk from the NCBI FTP server as `.tar.gz` archives:

```bash
python scripts/preprocessing/download_data.py
python scripts/preprocessing/extract_data.py
```

Each XML article is then parsed into a structured JSONL record containing: `id`, `title`, `keywords`, `abstract`, `introduction`, `results`, `discussion`, and `conclusion`:

```bash
python scripts/parser.py
```

Preprocessing filters out records with blank titles or abstracts and applies text cleaning:

```bash
python scripts/preprocessing/0_remove_blank_title_abstract.py
python scripts/preprocessing/2_text_cleaning.py
```

---

### 2. Embedding & Vector Indexing

[src/embedding/pipeline.py](src/embedding/pipeline.py) implements the full ingestion pipeline:

- **Embedding model**: `intfloat/e5-base-v2` (768-dimensional, cosine similarity)
- **Chunking strategy**: Sentence-based chunking with a 500-token limit using NLTK's `sent_tokenize`. Each chunk includes title, keywords, and abstract text. Chunks use the `passage:` prefix per the E5 model's instruction-following convention.
- **Vector database**: [Qdrant](https://qdrant.tech), collection `pmc_e5_base_sentence_based`
- **Batch processing**: Configurable batch sizes with resumable progress (already-processed paper IDs are logged to avoid re-embedding on restart)
- **Stored payload per vector**: `title`, `keywords`, `chunk_text`, `full_abstract`, `introduction`, `results`, `discussion`, `conclusion`

```bash
python src/embedding/pipeline.py
```

---

### 3. Answer Generation

All four conditions use the same input dataset (`pqa_labeled.jsonl` from PubMedQA) and share the same inference parameters: `temperature=0.1`, `top_p=0.95`, `max_new_tokens=300`, `repeat_penalty=1.1`.

#### Vanilla LLM
Queries Mistral-7B-Instruct via the Ollama local API with a direct medical expert prompt — no external context.

```bash
python src/generation/vanilla/vanilla_llm.py <input.jsonl> <output.jsonl> --model mistral:instruct
```

#### RAG
At inference time, each question is encoded with `intfloat/e5-base-v2` (using a `query:` prefix) and used to retrieve the top-k most semantically similar chunks from Qdrant. The retrieved context (title, abstract, results) is injected into the prompt before querying Mistral via Ollama.

```bash
python src/generation/rag/rag_llm.py <input.jsonl> <output.jsonl> \
  --model mistral:instruct \
  --collection pmc_e5_base_sentence_based \
  --top-k 1
```

#### Fine-Tuned LLM
Loads the fine-tuned `girayzkrt/Mist-FT-2106-4bit` checkpoint directly via HuggingFace `transformers`. Uses 4-bit quantization and runs on CUDA with `torch.float16`.

```bash
python src/generation/fine-tuned/fine_tuned_llm.py <input.jsonl> <output.jsonl> \
  --model girayzkrt/Mist-FT-2106-4bit
```

#### Fine-Tuned + RAG
Combines the fine-tuned model with Qdrant retrieval. The fine-tuned model receives retrieved context (abstract + results sections) alongside the question via its chat template.

```bash
python src/generation/fine_rag/fine_and_rag.py <input.jsonl> <output.jsonl> \
  --model girayzkrt/Mist-FT-2106-4bit \
  --collection pmc_e5_base_sentence_based \
  --top-k 1
```

---

### 4. Evaluation

#### BERTScore
[src/evaluation/bertscore/get_bert.py](src/evaluation/bertscore/get_bert.py) computes precision, recall, and F1 against ground-truth answers using `microsoft/deberta-xlarge-mnli` as the reference model. Results are broken down by question type and saved to CSV.

#### Semantic Similarity + ROUGE-L
[src/evaluation/semantic_similarity_and_rouge_l/semantic_scoring.py](src/evaluation/semantic_similarity_and_rouge_l/semantic_scoring.py) computes:
- **Semantic Similarity**: Cosine similarity between `intfloat/e5-large-v2` embeddings of the generated and ground-truth answer
- **ROUGE-L F1**: Longest common subsequence-based overlap

The script compares all four conditions side-by-side, computes improvements over the vanilla baseline, and generates publication-ready visualizations (bar charts + box plots).

```python
from src.evaluation.semantic_similarity_and_rouge_l.semantic_scoring import SemanticSimilarity

benchmark = SemanticSimilarity()
results = benchmark.compare_approaches(
    no_rag_file='vanilla_results.jsonl',
    with_rag_file='rag_results.jsonl',
    fine_tuned_file='fine_results.jsonl',
    fine_tuned_rag_file='fine_rag_results.jsonl'
)
benchmark.create_visualizations(results)
```

---

## Fine-Tuned Model

The fine-tuned model is publicly available on HuggingFace:

**[girayzkrt/Mist-FT-2106-4bit](https://huggingface.co/girayzkrt/Mist-FT-2106-4bit)**

It is a 4-bit quantized Mistral-7B-Instruct fine-tuned on biomedical literature from PubMed Central. It can be loaded directly with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "girayzkrt/Mist-FT-2106-4bit",
    device_map="cuda",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("girayzkrt/Mist-FT-2106-4bit")
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base LLM | Mistral-7B-Instruct (via Ollama) |
| Fine-tuned LLM | `girayzkrt/Mist-FT-2106-4bit` (HuggingFace) |
| Embedding model (retrieval) | `intfloat/e5-base-v2` |
| Embedding model (evaluation) | `intfloat/e5-large-v2` |
| Vector database | Qdrant |
| BERTScore reference model | `microsoft/deberta-xlarge-mnli` |
| Data source | PubMed Central (PMC) Open Access |
| Evaluation dataset | PubMedQA (labeled) |
| Deep learning framework | PyTorch + HuggingFace Transformers |

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies include: `torch`, `transformers`, `sentence-transformers`, `qdrant-client`, `evaluate`, `rouge-score`, `nltk`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`, `beautifulsoup4`.

Running the fine-tuned model requires a **CUDA-capable GPU** with sufficient VRAM (recommended: 8GB+). The embedding pipeline also benefits from GPU acceleration.

---

## Research Paper

The full thesis document is available in this repository:

[BSc_Thesis__Giray_Kurt.pdf](./BSc_Thesis__Giray_Kurt.pdf)

---

## Author

**Giray Kurt** - [@Girayzkrt](https://github.com/Girayzkrt)
