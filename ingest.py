# ── IMPORTS ───────────────────────────────────────────────────────────────────
import os
import pickle
import numpy as np
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
PDF_PATH      = os.path.join(os.path.dirname(__file__), "/Users/suheetsonawane/Desktop/India Vapsi/Github/Projects/Apple Sales RAG LLM/10K.pdf")
STORE_DIR     = os.path.join(os.path.dirname(__file__), "vector_store")
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100


# ── PHASE 1: EXTRACT TEXT FROM PDF ────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    pages = []
    print(f"📄 Opening PDF: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"   Total pages: {total}")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text and text.strip():
                text = text.replace("\x00", "")
                text = " ".join(text.split())

                pages.append({
                    "page":   i + 1,
                    "text":   text,
                    "source": f"Apple 10-K FY2025 — Page {i + 1}"
                })

    print(f"   ✅ Extracted {len(pages)} / {total} pages\n")
    return pages


# ── PHASE 2A: SPLIT PAGES INTO CHUNKS ─────────────────────────────────────────
def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for j, split in enumerate(splits):
            chunks.append({
                "text":   split,
                "page":   page["page"],
                "source": page["source"],
                "chunk":  j + 1
            })

    avg = sum(len(c["text"]) for c in chunks) // len(chunks)
    print(f"✂️  {len(pages)} pages → {len(chunks)} chunks (avg {avg} chars)\n")
    return chunks


# ── PHASE 2B: VECTOR STORE (LOCAL TF-IDF) ─────────────────────────────────────
class LocalVectorStore:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.matrix = None
        self.chunks = []

    def add_chunks(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)
        print(f"🔢 {len(chunks)} chunks embedded | vocab: {len(self.vectorizer.vocabulary_)} terms\n")

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        q_vec  = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_k  = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top_k if scores[i] > 0]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "store.pkl"), "wb") as f:
            pickle.dump(self, f)
        print(f"💾 Saved to {path}/store.pkl\n")

    @staticmethod
    def load(path: str) -> "LocalVectorStore":
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            return pickle.load(f)


# ── MAC VERSION: HUGGINGFACE SEMANTIC EMBEDDINGS ───────────────────────────────
def build_huggingface_store(chunks: list[dict]):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    texts     = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"], "source": c["source"]} for c in chunks]

    store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=os.path.join(STORE_DIR, "chroma_db")
    )

    print(f"✅ ChromaDB saved to {STORE_DIR}/chroma_db\n")
    return store


# ── SANITY CHECK: TEST RETRIEVAL BEFORE BUILDING THE APP ──────────────────────
def test_retrieval(store):
    print("🧪 Retrieval test:\n")
    queries = [
        "What was Apple's total revenue in 2025?",
        "How did iPhone sales perform?",
        "What risks did Apple mention about AI?",
        "Which geographic region had the highest sales?"
    ]
    for q in queries:
        print(f"Q: {q}")
        results = store.similarity_search(q, k=2)
        for r in results:
            print(f"   [{r.metadata.get('source', '')}] {r.page_content[:180]}...")
        print()


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Apple 10-K RAG — Ingestion Pipeline")
    print("=" * 60 + "\n")

    pages  = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_pages(pages)
    
    
    # LOCAL VERSION — works offline, no internet needed
    # store = LocalVectorStore()
    # store.add_chunks(chunks)
    # store.save(STORE_DIR)

    # MAC VERSION — uncomment for true semantic embeddings
    store = build_huggingface_store(chunks)

    test_retrieval(store)

    print("✅ Done — run:  streamlit run app.py")