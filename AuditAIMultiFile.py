import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os

# OpenAI API Key
OPENAI_API_KEY = "sk-wzADUHZAY9TDphuTQZfYQA"

st.set_page_config(page_title="Audit & Compliance Agent", layout="wide")
st.title("📑 Audit & Compliance Document Summarization Agent")

# Document Upload Section
with st.sidebar:
    st.header("⚙️ Documents")
    st.title("Upload Audit & Compliance Documents")
    file = st.file_uploader(
        "Upload documents (PDF, DOCX, XLSX) and start asking questions",
        type=["pdf", "docx", "xls", "xlsx"]
    )

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    full_text = ""
    for para in doc.paragraphs:
        full_text += para.text + "\n"
    return full_text

# Function to extract text from Excel
def extract_text_from_excel(file):
    df = pd.read_excel(file, sheet_name=None)  # read all sheets
    text = ""
    for sheet_name, sheet_df in df.items():
        text += f"Sheet: {sheet_name}\n"
        text += sheet_df.astype(str).to_string(index=False)
        text += "\n\n"
    return text

# Extract the data
if file is not None:
    file_type = file.name.split(".")[-1].lower()
    text = ""

    if file_type == "pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file_type == "docx":
        text = extract_text_from_docx(file)
    elif file_type in ["xls", "xlsx"]:
        text = extract_text_from_excel(file)
    else:
        st.error("Unsupported file type!")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings & FAISS store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Initialize LLM
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        max_tokens=1000,
        model_name="gpt-3.5-turbo"
    )

    # Setup RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff"
    )

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Summary", "⚠️ Risks", "💬 Q&A"])

    with tab1:
        st.subheader("Document Summary")
        summary_query = "Summarize the key findings, compliance issues, and recommendations."
        st.write(qa.run(summary_query))

    with tab2:
        st.subheader("Risk Highlights")
        risk_query = "List all compliance risks with severity (High/Medium/Low) and rationale."
        st.write(qa.run(risk_query))

    with tab3:
        st.subheader("Ask a Question")
        user_q = st.text_input("Enter your query:")
        if user_q:
            st.write(qa.run(user_q))
