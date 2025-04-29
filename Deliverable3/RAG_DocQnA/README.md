# Mistral Document Q&A App

## Date: 26 April, 2025  
## Author: Siddharth Kori

This is a Streamlit-based application that allows users to ask questions about a collection of PDF documents. It uses the Mistral LLM for answering questions based on content extracted and embedded from PDFs using FAISS and HuggingFace sentence-transformers.

---

## Features

- Ask questions based only on the content of uploaded PDFs
- Uses LangChain's Retrieval-Augmented Generation (RAG) pipeline
- Loads and splits documents into searchable chunks
- Fast vector search using FAISS
- Embedding powered by HuggingFace
- Mistral API (mistral-small-latest model)

---

## Tech Stack

- Python
- Streamlit
- LangChain
- MistralAI API
- FAISS
- HuggingFace Embeddings
- PyPDFDirectoryLoader

---

## Directory Structure

# Demonstrating RAG (Retrieval-Augmented Generation) with Mistral Document Q&A

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is an architecture that combines:

- **Retrieval**: Finding relevant documents or context from a large external source.
- **Augmentation**: Injecting that context into a prompt.
- **Generation**: Using a language model to generate answers based on the injected context.

---

## How This App Demonstrates RAG

This Streamlit-based Q&A app implements RAG using LangChain, Mistral LLM, and FAISS vector store.

---

## Step-by-Step Breakdown

### 1. **Document Loading**

```python
from langchain.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("us_census")
docs = loader.load()
```

### 2. **Document Chunking**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs[:20])
```

### 3. **Embedding the Chunks**

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
st.session_state.vectors = FAISS.from_documents(chunks, embeddings)
```

### 4. **Context Retrieval from Vector Store**

```python
retriever = st.session_state.vectors.as_retriever()
```

### 5. **Prompt Construction with Retrieved Context**

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context only:
<context>
{context}
</context>

Question: {input}
""")
```
### 6. Prompt Construction with Retrieved Context

```python
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context only:
<context>
{context}
</context>

Question: {input}
""")
```

### 7. **Generation with Mistral LLM**

```python
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain

doc_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, doc_chain)
response = retrieval_chain.invoke({"input": question})
```


## Benefits of Using RAG

- **Grounded Answers**: Reduces hallucinations by using real document data.

- **Modular Design**: Easy to scale and maintain.

- **Transparent Reasoning**: Shows which parts of the data were used to generate the answer.