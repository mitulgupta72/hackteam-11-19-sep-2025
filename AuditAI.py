import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os

#Open Key OPENAI
OPENAI_API_KEY="sk-wzADUHZAY9TDphuTQZfYQA"

st.set_page_config(page_title="Audit & Compliance Agent", layout="wide")

st.title("📑 Audit & Compliance Document Summarization Agent")

# Document Upload Section
with st.sidebar:
    st.header("⚙️ Documents")
    st.title("Upload Audit & Compliance Documents")
    # File upload
    file = st.file_uploader("Upload Audit documents and start asking questions", type=["pdf", "docx"])

#Extract the data
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    # Create embeddings & FAISS store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    #vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user input
    #user_question = st.text_input("Type your question here")

    #llm = ChatOpenAI(
    #        openai_api_key=OPENAI_API_KEY,
    #        temperature=0,
    #        max_tokens=1000,
    #        model_name="gpt-3.5-turbo"
    #    )

    # Use RetrievalQA
    #qa = RetrievalQA.from_chain_type(
    #        llm=llm,
    #        retriever=vector_store.as_retriever(),
    #        chain_type="stuff"
    #    )

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Summary", "⚠️ Risks", "💬 Q&A"])

    with tab1:
        st.subheader("Document Summary")
        summary_query = "Summarize the key findings, compliance issues, and recommendations."
        #st.write(qa.run(summary_query))

    with tab2:
        st.subheader("Risk Highlights")
        risk_query = "List all compliance risks with severity (High/Medium/Low) and rationale."
        #st.write(qa.run(risk_query))

    with tab3:
        st.subheader("Ask a Question")
        user_q = st.text_input("Enter your query:")
        if user_q:
            #st.write(qa.run(user_q))
            print('test')

    # Cleanup
    #os.remove(file_path)
    
