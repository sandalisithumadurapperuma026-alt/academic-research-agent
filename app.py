import streamlit as st
import os

st.set_page_config(page_title="Academic Research Agent", layout="wide")

st.title("📚 Academic Research Agent")
st.subheader("Upload your research paper (PDF) and ask questions!")

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    
    # Save uploaded file locally
    os.makedirs("./data", exist_ok=True)
    pdf_path = os.path.join("./data", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

# Chat / Question Input Section
user_query = st.text_input("Ask a question about your research paper:")

if user_query:
    st.write(f"**Your Question:** {user_query}")
    st.info("Agent processing... (Connecting RAG & LangGraph next)")