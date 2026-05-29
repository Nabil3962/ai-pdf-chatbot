import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.title("📄 AI PDF Chatbot (LangChain + RAG)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    # Save file
    file_path = os.path.join("temp.pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Vector DB
    db = Chroma.from_documents(chunks, embeddings)

    retriever = db.as_retriever()

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key
    )

    # RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    st.subheader("Ask questions from your PDF")

    question = st.text_input("Enter your question")

    if question:
        result = qa.invoke(question)
        st.write("### Answer:")
        st.write(result["result"])
