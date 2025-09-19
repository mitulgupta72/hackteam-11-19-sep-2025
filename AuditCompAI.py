import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import tempfile

st.set_page_config(page_title="Audit & Compliance Agent", layout="wide")

st.title("📑 Audit & Compliance Document Summarization Agent")

# Sidebar
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Retriever Top-K", 2, 10, 4)

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load and process
    loader = PyPDFLoader(uploaded_file)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # Embeddings + Vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Setup QA chain
    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Tabs for Summary, Risks, QA
    tab1, tab2, tab3 = st.tabs(["📋 Summary", "⚠️ Risks", "💬 Q&A"])

    with tab1:
        st.subheader("Document Summary")
        query = "Summarize the key findings, compliance issues, and recommendations."
        response = qa.run(query)
        st.write(response)

    with tab2:
        st.subheader("Risk Highlights")
        query = "Extract all compliance risks, with severity (High/Medium/Low) and rationale."
        response = qa.run(query)
        st.write(response)

    with tab3:
        st.subheader("Ask a Question")
        user_q = st.text_input("Enter your query:")
        if user_q:
            response = qa.run(user_q)
            st.write(response)
