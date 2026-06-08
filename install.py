"""
=============================================================
  APPLE 10-K RAG — install.py
=============================================================
  What this file does:
    Installs every Python library the project needs.
    Run this ONCE before anything else.

  Run it with:
    python install.py

  What it installs and WHY each one is needed:
=============================================================
"""

import subprocess
import sys

# ── HELPER ────────────────────────────────────────────────────────────────────

def install(package: str, reason: str):
    """
    Installs a single package using pip and prints what it is for.
    """
    print(f"\n📦 Installing: {package}")
    print(f"   Why: {reason}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package, "--quiet"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"   ✅ Done")
    else:
        print(f"   ❌ Failed — error below:")
        print(result.stderr)


# ── PACKAGES ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Apple 10-K RAG — Installing Prerequisites")
print("=" * 60)

install(
    "pdfplumber",
    "Reads PDF files and extracts human-readable text page by page. "
    "Without this, Python sees the PDF as raw binary data, not words."
)

install(
    "langchain-text-splitters",
    "Provides RecursiveCharacterTextSplitter — splits long pages of text "
    "into small overlapping chunks. Part of the LangChain ecosystem."
)

install(
    "langchain-community",
    "LangChain's community integrations — includes the ChromaDB vector "
    "store wrapper we use in the Mac/HuggingFace version of the pipeline."
)

install(
    "langchain-huggingface",
    "Connects LangChain to HuggingFace embedding models. "
    "Used in the Mac version of ingest.py to load all-MiniLM-L6-v2 "
    "for true semantic (meaning-based) search."
)

install(
    "sentence-transformers",
    "The actual embedding model library. all-MiniLM-L6-v2 runs through "
    "this. It converts text into 384-dimensional vectors that capture "
    "meaning. Trained on 1 billion sentence pairs."
)

install(
    "chromadb",
    "Local vector database — stores the 776 chunk embeddings on disk "
    "so you only need to run ingest.py once. Used in the Mac version. "
    "Very fast at similarity search even with millions of vectors."
)

install(
    "scikit-learn",
    "Provides TfidfVectorizer and cosine_similarity — used in the local "
    "TF-IDF version of the vector store (no internet required). "
    "Also useful for future ML projects in your portfolio."
)

install(
    "numpy",
    "Fast math library — we use np.argsort() to rank similarity scores "
    "and find the top-k most relevant chunks. Used everywhere in data science."
)

install(
    "streamlit",
    "Turns your Python script into a web app with zero HTML/CSS. "
    "Provides the chat interface, sidebar, expanders, and buttons "
    "in app.py. Deploy to Streamlit Cloud for free with a public URL."
)

install(
    "groq",
    "Official Groq API client — connects to LLaMA 3 running on Groq's "
    "hardware. Free tier available at console.groq.com. "
    "Used in app.py to generate answers from the retrieved chunks."
)

install(
    "langchain-groq",
    "LangChain's Groq integration — alternative way to call Groq inside "
    "LangChain chains. Useful when you build the full LangChain RAG chain "
    "in Phase 3 of the project."
)

# ── VERIFY ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Verifying installations...")
print("=" * 60)

packages_to_check = [
    ("pdfplumber",              "pdfplumber"),
    ("langchain_text_splitters","langchain-text-splitters"),
    ("langchain_community",     "langchain-community"),
    ("sklearn",                 "scikit-learn"),
    ("numpy",                   "numpy"),
    ("streamlit",               "streamlit"),
    ("groq",                    "groq"),
    ("chromadb",                "chromadb"),
]

all_good = True
for import_name, package_name in packages_to_check:
    try:
        __import__(import_name)
        print(f"   ✅ {package_name}")
    except ImportError:
        print(f"   ❌ {package_name} — try: pip install {package_name}")
        all_good = False

# sentence-transformers check separately (import name is different)
try:
    from sentence_transformers import SentenceTransformer
    print(f"   ✅ sentence-transformers")
except ImportError:
    print(f"   ❌ sentence-transformers — try: pip install sentence-transformers")
    all_good = False

# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)

if all_good:
    print("  🎉 All packages installed successfully!")
    print("\n  Next steps:")
    print("  1. Put your Apple 10-K PDF in a folder called  data/")
    print("     and rename it  10K.pdf")
    print("  2. Get a free Groq API key at  console.groq.com")
    print("  3. Run:  python ingest.py")
    print("  4. Run:  streamlit run app.py")
else:
    print("  ⚠️  Some packages failed. Try running manually:")
    print("  pip install pdfplumber langchain-text-splitters langchain-community")
    print("  pip install langchain-huggingface sentence-transformers chromadb")
    print("  pip install scikit-learn numpy streamlit groq langchain-groq")

print("=" * 60)
