import os
import glob
import warnings
import logging
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["LANGCHAIN_DEPRECATION_WARNINGS"] = "False"

from pypdf import PdfReader
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.agents import process_research_query, generate_novel_research_paper


st.set_page_config(
    page_title="Academic Research Assistant", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    /* Global Background and Text Color */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }

    /* Main Title Styling */
    .main-title {
        font-size: 2.2rem;
        color: #38bdf8;
        font-weight: 700;
        margin-bottom: 0rem;
    }
    .sub-caption {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    
    /* Modern Button Styling with Hover Glow Effect */
    div.stButton > button {
        background-color: #1e293b;
        color: #f8fafc;
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid #334155;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #0284c7;
        border-color: #38bdf8;
        color: #ffffff;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }

    /* Sidebar Background Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }

    /* Chat Input Bar Styling */
    .stChatInputContainer {
        border-radius: 12px;
        border: 1px solid #334155;
        background-color: #1e293b;
    }
    </style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "generated_paper" not in st.session_state:
    st.session_state.generated_paper = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


@st.cache_resource
def get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_domain_corpus():
    data_text = ""
    pdf_files = glob.glob("data/*.pdf")
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    data_text += extracted + "\n"
        except Exception:
            pass
    return data_text

@st.cache_resource
def get_vector_store():
    corpus_text = load_domain_corpus()
    if corpus_text.strip():
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = text_splitter.split_text(corpus_text)
        return FAISS.from_texts(texts=chunks, embedding=get_embeddings())
    return None

if st.session_state.vector_store is None:
    st.session_state.vector_store = get_vector_store()

def create_pdf(text_content):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    custom_body = ParagraphStyle('CustomBody', parent=styles['BodyText'], fontSize=10, leading=14, spaceAfter=8)
    custom_title = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=14, leading=18, spaceAfter=12)
    
    story = []
    lines = text_content.split('\n')
    for line in lines:
        clean_line = line.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if not clean_line:
            story.append(Spacer(1, 6))
            continue
            
        if clean_line.startswith('#'):
            clean_line = clean_line.lstrip('#').strip()
            story.append(Paragraph(f"<b>{clean_line}</b>", custom_title))
        else:
            clean_line = clean_line.replace('**', '<b>', 1).replace('**', '</b>', 1)
            story.append(Paragraph(clean_line, custom_body))
            
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
    st.markdown("### **Research Control Panel**")
    st.caption("Multi-Agent RAG System")
    
    pdf_count = len(glob.glob("data/*.pdf"))
    st.success(f"📚 **Corpus Status:** {pdf_count} PDFs Loaded")
    
    with st.expander("📤 Upload Custom PDF"):
        uploaded_file = st.file_uploader("Add research paper", type=["pdf"], label_visibility="collapsed")
        if uploaded_file is not None:
            if "processed_files" not in st.session_state:
                st.session_state.processed_files = set()
                
            if uploaded_file.name not in st.session_state.processed_files:
                try:
                    with st.spinner("Indexing file..."):
                        reader = PdfReader(uploaded_file)
                        new_text = ""
                        for page in reader.pages:
                            new_text += page.extract_text() or ""
                        
                        if new_text.strip():
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
                            new_chunks = text_splitter.split_text(new_text)
                            
                            if st.session_state.vector_store is not None:
                                st.session_state.vector_store.add_texts(new_chunks)
                            else:
                                st.session_state.vector_store = FAISS.from_texts(texts=new_chunks, embedding=get_embeddings())
                                
                            st.session_state.processed_files.add(uploaded_file.name)
                            st.success("✅ Added Successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("⚙️ Agent Actions")
    
    if st.button("✨ Generate Corpus Summary", use_container_width=True):
        if not st.session_state.vector_store:
            st.warning("No corpus available!")
        else:
            with st.spinner("Agents analyzing corpus..."):
                try:
                    summary_prompt = "Provide a comprehensive structured summary of the research paper corpus."
                    docs = st.session_state.vector_store.similarity_search(summary_prompt, k=3)
                    retrieved_context = "\n\n".join([doc.page_content for doc in docs])[:3000]
                    
                    summary_res = process_research_query(retrieved_context, summary_prompt)
                    st.session_state.messages.append({"role": "assistant", "content": f"### 📊 Corpus Summary:\n\n{summary_res}"})
                    st.rerun()
                except Exception as err:
                    st.error(f"Error: {err}")

    if st.button("📄 Draft Novel Research Paper", use_container_width=True):
        if not st.session_state.vector_store:
            st.warning("No corpus available!")
        else:
            with st.spinner("Agents synthesizing and drafting paper..."):
                try:
                    docs = st.session_state.vector_store.similarity_search("methodology findings limitations gap", k=4)
                    retrieved_context = "\n\n".join([doc.page_content for doc in docs])[:4000]
                    
                    new_paper = generate_novel_research_paper(retrieved_context)
                    st.session_state.generated_paper = new_paper
                    st.session_state.messages.append({"role": "assistant", "content": f"### 📄 Drafted Research Paper:\n\n{new_paper}"})
                    st.rerun()
                except Exception as err:
                    st.error(f"Error: {err}")

    if st.session_state.generated_paper:
        st.markdown("---")
        try:
            pdf_data = create_pdf(st.session_state.generated_paper)
            st.download_button(
                label="⬇️ Download Paper as PDF",
                data=pdf_data,
                file_name="Research_Paper_Draft.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as pdf_err:
            st.error(f"PDF error: {pdf_err}")

    
    st.markdown("---")
    st.subheader("📜 Recent Chat History")
    
    if st.session_state.messages:
        for idx, msg in enumerate(st.session_state.messages[-6:]):
            content = msg["content"]
            first_line = content.split('\n')[0].replace('#', '').strip()
            short_title = (first_line[:22] + '...') if len(first_line) > 22 else first_line
            
            if msg["role"] == "user":
                st.markdown(f"💬 **User:** {short_title}")
            else:
                st.markdown(f"🤖 **AI:** {short_title}")
    else:
        st.caption("No history recorded yet.")

    st.markdown("---")
    if st.button("🗑️ Clear Workspace", use_container_width=True):
        st.session_state.messages = []
        st.session_state.generated_paper = ""
        st.rerun()


st.markdown('<p class="main-title">🧠 Academic Research Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-caption">Powered by LangGraph, Groq Multi-Agent Architecture & RAG Pipeline</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Interactive Research Chat", "📄 Generated Paper View"])

with tab1:
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_input := st.chat_input("Ask a question about your research paper corpus..."):
        if not st.session_state.vector_store:
            st.warning("Please ensure documents are loaded in the data folder!")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Multi-Agent workflow processing..."):
                    try:
                        docs = st.session_state.vector_store.similarity_search(user_input, k=3)
                        retrieved_context = "\n\n".join([doc.page_content for doc in docs])[:3000]
                        
                        response = process_research_query(retrieved_context, user_input)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as err:
                        st.error(f"Error occurred: {err}")

with tab2:
    st.subheader("📝 Live Paper Draft Preview")
    if st.session_state.generated_paper:
        st.markdown(st.session_state.generated_paper)
    else:
        st.info("No research paper drafted yet. Click **'Draft Novel Research Paper'** on the sidebar to generate one.")