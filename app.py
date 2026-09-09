# ── IMPORTS ───────────────────────────────────────────────────────────────────
import os
import streamlit as st
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apple 10-K Analyst",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Apple 10-K Intelligence")
st.caption("Ask anything about Apple's FY2025 Annual Report — powered by RAG")


# ── LOAD CHROMADB (auto-builds if not found) ──────────────────────────────────
@st.cache_resource
def load_store():
    import tempfile
    
    chroma_path = "/tmp/chroma_db"
    pdf_path = os.path.join(os.path.dirname(__file__), "10K.pdf")
    
    # Debug info
    st.write(f"PDF exists: {os.path.exists(pdf_path)}")
    st.write(f"ChromaDB exists: {os.path.exists(chroma_path)}")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(chroma_path):
        return Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    
    # Build from PDF
    if not os.path.exists(pdf_path):
        st.error(f"PDF not found at: {pdf_path}")
        return None
        
    with pdfplumber.open(pdf_path) as pdf:
        pages = [{"text": p.extract_text() or "", "page": i+1} for i, p in enumerate(pdf.pages)]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            from langchain_core.documents import Document
            docs.append(Document(page_content=chunk, metadata={"source": f"Page {page['page']}"}))
    
    store = Chroma.from_documents(docs, embeddings, persist_directory=chroma_path)
    return store


store = load_store()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    groq_api_key = os.environ.get("GROQ_API_KEY", "") or st.text_input("Groq API Key", type="password", help="Get free key at console.groq.com")
    
    
    st.markdown("---")
    st.subheader("📄 Data Source")
    st.success("Apple 10-K FY2025 — 80 pages, 776 chunks")

    st.markdown("---")
    st.subheader("💡 Sample Questions")
    for q in [
        "What was Apple's total revenue in 2025?",
        "How did Services revenue grow year over year?",
        "Which geographic segment had the highest sales?",
        "What risks did Apple highlight about AI?",
        "How much did Apple spend on R&D?",
        "What was Apple's net income in 2025?",
        "How did Mac sales perform?",
        "What is Apple's cash position?",
    ]:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["prefill"] = q


# ── CHAT HISTORY ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.caption(f"**{s['source']}** — relevance: {s['score']:.3f}")
                    st.markdown(f"> {s['text'][:300]}...")


# ── USER INPUT ────────────────────────────────────────────────────────────────
prefill    = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask about Apple's financials, strategy, or risks...")
question   = prefill or user_input


# ── RAG FLOW ──────────────────────────────────────────────────────────────────
if question:

    # show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # retrieve top 8 relevant chunks from ChromaDB
    raw     = store.similarity_search_with_relevance_scores(question, k=8)
    results = [
        {
            "text":   doc.page_content,
            "source": doc.metadata.get("source", ""),
            "page":   doc.metadata.get("page", ""),
            "score":  round(score, 3)
        }
        for doc, score in raw
    ]

    # no api key — show raw chunks
    if not groq_api_key:
        answer = "⚠️ Add your Groq API key in the sidebar to get AI answers.\n\n**Raw retrieved context:**\n\n"
        for r in results[:3]:
            answer += f"**{r['source']}:**\n{r['text'][:400]}\n\n---\n"

    # with api key — call llama 3 with retrieved context
    else:
        context = "\n\n---\n\n".join([f"[{r['source']}]\n{r['text']}" for r in results])

        system_prompt = """You are a financial analyst assistant for Apple Inc.
Answer ONLY using the context provided from Apple's official FY2025 10-K filing.
Be precise with numbers. State which fiscal year figures refer to.
If the exact answer is not in the context, summarise what the context does say."""

        user_prompt = f"""Context from Apple's FY2025 10-K:
{context}

Question: {question}

Answer based only on the context above:"""

        try:
            response = Groq(api_key=groq_api_key).chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            answer = response.choices[0].message.content

        except Exception as e:
            answer  = f"❌ Groq error: {str(e)}"
            results = []

    # display answer and sources
    with st.chat_message("assistant"):
        st.markdown(answer)
        if results:
            with st.expander("📎 Sources"):
                for s in results:
                    st.caption(f"**{s['source']}** — relevance: {s['score']:.3f}")
                    st.markdown(f"> {s['text'][:300]}...")

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": results
    })
