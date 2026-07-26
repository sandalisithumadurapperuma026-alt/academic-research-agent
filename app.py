import streamlit as st
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.agents import process_research_query

st.set_page_config(page_title="Academic Research Assistant (Agentic AI)", layout="wide", page_icon="🧠")

st.title("🧠 Academic Research Assistant")
st.caption("Multi-Agent Architecture powered by LangGraph, Groq Models & RAG")

# State Management
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Embeddings Setup
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sidebar UI
with st.sidebar:
    st.header("📑 Document Hub")
    st.write("Upload & manage research papers")
    
    uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])
    
    if uploaded_file is not None:
        try:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            
            if text.strip() and st.session_state.pdf_text != text:
                st.session_state.pdf_text = text
                # Build FAISS Vector Store
                with st.spinner("Indexing PDF into FAISS Vector Database..."):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
                    chunks = text_splitter.split_text(text)
                    st.session_state.vector_store = FAISS.from_texts(texts=chunks, embedding=get_embeddings())
                    st.session_state.messages = []
                st.success("✅ RAG Vector Database Ready!")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

    st.markdown("---")
    st.subheader("🛠️ Quick Tools")
    
    # Generate Summary Button
    if st.button("✨ Generate Summary", use_container_width=True):
        if not st.session_state.vector_store:
            st.warning("Please upload a research PDF first!")
        else:
            with st.spinner("Executing Multi-Agent Workflow..."):
                summary_prompt = "Provide a comprehensive structured summary of the research paper."
                # Retrieve top 3 chunks (limited to 3000 chars to respect Rate Limits)
                docs = st.session_state.vector_store.similarity_search(summary_prompt, k=3)
                retrieved_context = "\n\n".join([doc.page_content for doc in docs])[:3000]
                
                summary_res = process_research_query(retrieved_context, summary_prompt)
                st.session_state.messages.append({"role": "assistant", "content": summary_res})
                st.rerun()

    # Clear Chat History Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Chat Input Box
if user_input := st.chat_input("Ask a question about your research paper..."):
    if not st.session_state.vector_store:
        st.warning("Please upload a PDF paper first!")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("RAG Searching & Multi-Agent Reasoning..."):
                # Retrieve top 3 chunks (limited to 3000 chars to respect Rate Limits)
                docs = st.session_state.vector_store.similarity_search(user_input, k=3)
                retrieved_context = "\n\n".join([doc.page_content for doc in docs])[:3000]
                
                # Run Multi-Agent Execution
                response = process_research_query(retrieved_context, user_input)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})