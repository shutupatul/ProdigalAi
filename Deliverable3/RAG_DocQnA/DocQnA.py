#26 April,2025
#Siddlharth Kori

import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.chat_models.mistralai import ChatMistralAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

st.title("📄 Mistral Document Q&A")

# Mistral Model Setup
llm = ChatMistralAI(
    api_key=mistral_api_key,
    model="mistral-small-latest",  # or "mistral-medium-latest"
    temperature=0.7
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context only:
<context>
{context}
</context>

Question: {input}
""")

# Vector Store Creation
def create_vector_store():
    if "vectors" not in st.session_state:
        loader = PyPDFDirectoryLoader("us_census")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs[:20])

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectors = FAISS.from_documents(chunks, embeddings)

        st.success("✅ Vector store created!")


# Input
question = st.text_input("Ask a question based on the documents:")

# Button to embed documents
if st.button("Embed Documents"):
    create_vector_store()

# Run Retrieval Chain
if question and "vectors" in st.session_state:
    with st.spinner("Thinking..."):
        retriever = st.session_state.vectors.as_retriever()
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": question})
        elapsed = time.process_time() - start

        st.subheader("📌 Answer")
        st.write(response['answer'])

        st.subheader("🧠 Top Matching Chunks")
        for i, doc in enumerate(response["context"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content)
            st.markdown("---")

        st.info(f"⏱️ Process time: {elapsed:.2f} seconds")
        