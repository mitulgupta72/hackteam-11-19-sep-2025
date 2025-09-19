import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="Audit & Compliance Agent (SLM)", layout="wide")
st.title("📑 Audit & Compliance Document Summarization Agent (SLM)")

# Upload file
with st.sidebar:
    file = st.file_uploader("Upload Audit/Compliance Document", type=["pdf", "docx"])

if file is not None:
    pdf_reader = PdfReader(file)
    text = "".join([page.extract_text() for page in pdf_reader.pages])

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)

    # Embeddings (MiniLM - small & fast)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # SLM (Mistral / Falcon / DistilGPT2)
    slm_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=slm_pipeline)

    # RetrievalQA
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever(), chain_type="stuff")

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