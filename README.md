# 🍎 Apple 10-K Intelligence — RAG-Powered Financial Analyst

A conversational AI that answers questions about Apple's FY2025 Annual Report using **Retrieval-Augmented Generation (RAG)**. Ask anything in plain English and get accurate, grounded answers pulled directly from Apple's official SEC filing — not hallucinated, not guessed.

> *"How much did Apple spend on R&D in 2025?"*
> → **"Apple spent $29,915 million on Research and Development in fiscal year 2025."**

---

## 🎯 What This Project Demonstrates

This project mirrors real-world AI/ML engineering work — the kind described in Apple's own AI/ML Engineer job postings:

- ✅ **RAG pipeline** — end-to-end retrieval-augmented generation from a real document
- ✅ **Semantic embeddings** — `all-MiniLM-L6-v2` via HuggingFace sentence-transformers
- ✅ **Vector database** — ChromaDB for fast similarity search
- ✅ **LLM integration** — LLaMA 3.3 70B via Groq API with grounded, citation-backed answers
- ✅ **Production UI** — Streamlit chat interface with source citations
- ✅ **Data source** — Apple's real FY2025 10-K SEC filing (80 pages)

---

## 🏗️ Architecture

```
Apple 10-K PDF (80 pages)
        │
        ▼
PDF Extraction — pdfplumber
        │
        ▼
Text Chunking — LangChain RecursiveCharacterTextSplitter
        │   chunk_size=500, overlap=100 → 776 chunks
        ▼
Semantic Embedding — all-MiniLM-L6-v2 (384-dim vectors)
        │
        ▼
Vector Store — ChromaDB (persisted to disk)
        │
        ▼  ◄── User asks a question
Similarity Search — top 8 semantically relevant chunks
        │
        ▼
Prompt = System Instructions + Retrieved Chunks + Question
        │
        ▼
LLaMA 3.3 70B via Groq API (temperature=0.1)
        │
        ▼
Grounded Answer + Source Citations in Streamlit UI
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| PDF Parsing | `pdfplumber` |
| Text Chunking | `LangChain RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector Database | `ChromaDB` |
| LLM | `LLaMA 3.3 70B` via `Groq API` |
| Orchestration | `LangChain` |
| UI | `Streamlit` |
| Language | `Python 3.11+` |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- A free Groq API key → [console.groq.com](https://console.groq.com)
- Apple FY2025 10-K PDF → [investor.apple.com](https://investor.apple.com/sec-filings/annual-reports/default.aspx)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/apple-10k-rag.git
cd apple-10k-rag

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Setup

```bash
# 4. Place the Apple 10-K PDF in the project root
# Rename it to: 10K.pdf

# 5. Run the ingestion pipeline (one-time setup, ~1 minute)
python3 ingest.py

# 6. Launch the app
streamlit run app.py
```

### Usage

1. Open the app in your browser (Streamlit will print the URL)
2. Paste your Groq API key in the sidebar
3. Ask any question about Apple's FY2025 financials

---

## 💬 Sample Questions

- *"What was Apple's total revenue in 2025?"*
- *"How did Services revenue grow year over year?"*
- *"How much did Apple spend on R&D?"*
- *"Which geographic segment had the highest sales?"*
- *"What risks did Apple highlight about AI?"*
- *"What was Apple's net income in 2025?"*
- *"How did Mac sales perform?"*
- *"What is Apple's cash position?"*

---

## 📁 Project Structure

```
apple-10k-rag/
├── app.py               # Streamlit chat interface
├── ingest.py            # PDF ingestion + vector store builder
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── 10K.pdf              # Apple FY2025 10-K (not tracked in git)
└── vector_store/
    └── chroma_db/       # ChromaDB vector store (auto-created)
```

---

## ⚙️ How It Works

### Phase 1 — Ingestion (`ingest.py`, run once)

1. **Extract** — `pdfplumber` reads all 80 pages of the 10-K PDF and extracts plain text
2. **Chunk** — `RecursiveCharacterTextSplitter` splits text into 776 overlapping chunks (500 chars each, 100 char overlap). Each chunk is tagged with its source page number
3. **Embed** — `all-MiniLM-L6-v2` converts each chunk into a 384-dimensional semantic vector
4. **Store** — All vectors are saved to a local `ChromaDB` database on disk

### Phase 2 — Query (`app.py`, every question)

1. **Input** — User types a question in the Streamlit chat interface
2. **Embed** — The question is converted to a vector using the same embedding model
3. **Retrieve** — ChromaDB finds the 8 most semantically similar chunks via cosine similarity
4. **Prompt** — Retrieved chunks are injected into a structured prompt with the question
5. **Generate** — LLaMA 3.3 70B reads the chunks and generates a grounded answer
6. **Cite** — The app displays the answer alongside source page citations

---

## 🔑 Key Design Decisions

**Why RAG instead of fine-tuning?**
RAG is faster to build, cheaper to run, and keeps the LLM's answers grounded in verifiable source data. Fine-tuning would bake the data into the model weights — making it harder to update and impossible to cite sources.

**Why semantic embeddings over keyword search?**
TF-IDF keyword matching fails when the user's phrasing differs from the document's language. Semantic embeddings understand that *"How much money did Apple make?"* and *"Apple's total net sales"* mean the same thing.

**Why temperature=0.1?**
Financial data requires precision. Low temperature keeps the LLM factual and deterministic rather than creative.

**Why chunk overlap?**
Overlapping chunks (100-char overlap) prevent answers from being split across chunk boundaries — ensuring complete context is always retrieved.

---

## 📦 requirements.txt

```
pdfplumber
langchain-text-splitters
langchain-community
langchain-huggingface
sentence-transformers
chromadb
scikit-learn
numpy
streamlit
groq
```

---

## 🗺️ Roadmap

- [ ] Add multiple filings (10-Q Q1/Q2/Q3 2025) for multi-year analysis
- [ ] Add conversational memory across turns
- [ ] Upgrade to re-ranking for better retrieval accuracy
- [ ] Deploy to Streamlit Cloud with public URL
- [ ] Add multi-agent workflow — one agent retrieves, one reasons, one answers

---

## 👤 Author

**Suheet Sonawane**
Post-Graduate Certification in AI & Big Data
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 Disclaimer

This project uses Apple's publicly available SEC filing for educational and portfolio purposes only. All financial data is sourced directly from Apple's official FY2025 Form 10-K.
