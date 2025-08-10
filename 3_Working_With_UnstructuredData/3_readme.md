# 🚀 RAG Unstructured Data Handling with LangChain

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Overview

Transform your messy, unstructured data into an intelligent, searchable knowledge base! This project demonstrates how to build powerful **Retrieval Augmented Generation (RAG) systems** using LangChain to process various document formats and extract meaningful insights.

### 🎯 What This Project Solves

- **📚 Document Chaos**: Turn scattered files into organized, searchable knowledge
- **🔍 Information Retrieval**: Find exact answers from vast document collections
- **💡 Intelligent Analysis**: Generate contextual responses from unstructured data
- **⚡ Automation**: Replace manual document searching with AI-powered retrieval

## 🛠️ Supported Document Types

| Format                 | Icon | Use Cases                           | Chunk Size Recommendation |
| ---------------------- | ---- | ----------------------------------- | ------------------------- |
| **Excel (.xlsx)**      | 📊   | Reviews, datasets, structured data  | 2000 chars                |
| **Word (.docx)**       | 📄   | Reports, articles, documentation    | 500 chars                 |
| **PowerPoint (.pptx)** | 📋   | Presentations, pitch decks          | 200-300 chars             |
| **EPUB (.epub)**       | 📖   | Ebooks, digital publications        | 300-400 chars             |
| **PDF (.pdf)**         | 📑   | Academic papers, contracts, manuals | 500-2200 chars            |

## 🏗️ Architecture

```mermaid
graph LR
    A[📁 Document Input] --> B[🔄 Document Loader]
    B --> C[✂️ Text Splitter]
    C --> D[🧠 Embeddings]
    D --> E[🗄️ Vector Store]
    E --> F[🔍 Similarity Search]
    F --> G[🤖 LLM Response]
```

## ⚙️ Core Workflow

### 1. 📥 Document Loading

```python
# Load documents with appropriate loaders
loader = UnstructuredExcelLoader("data.xlsx", mode="elements")
docs = loader.load()
```

### 2. ✂️ Text Chunking

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,    # Adjust based on document type
    chunk_overlap=200   # Maintain context between chunks
)
chunks = text_splitter.split_documents(docs)
```

### 3. 🧠 Embeddings Generation

```python
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"  # High performance model
)
```

### 4. 🗄️ Vector Storage

```python
db_faiss = FAISS.from_documents(chunks, embeddings)
```

### 5. 🔍 Query & Retrieval

```python
docs_faiss = db_faiss.similarity_search_with_score(query, k=5)
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install langchain-community
pip install langchain-openai
pip install faiss-cpu
pip install python-docx
pip install python-pptx
pip install pypandoc
pip install pymupdf
pip install unstructured[all-docs]
```

### Environment Setup

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
```

### Basic Usage Example

```python
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

def process_document(file_path, chunk_size=2000):
    # Load document
    loader = UnstructuredExcelLoader(file_path, mode="elements")
    docs = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db_faiss = FAISS.from_documents(chunks, embeddings)

    return db_faiss

def ask_question(db, query, k=5):
    # Retrieve relevant documents
    docs = db.similarity_search_with_score(query, k=k)

    # Prepare context
    context = "\n\n".join([doc.page_content for doc, _score in docs])

    # Generate response
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
    Based on this context: {context}
    Please answer this question: {query}
    If you don't know the answer, say you don't know.
    """

    response = model.invoke(prompt)
    return response.content
```

## 📋 Document-Specific Configurations

### 📊 Excel Files

- **Chunk Size**: 2000 characters
- **Best For**: Reviews, structured data, surveys
- **Mode**: `elements` (breaks into manageable parts)

### 📄 Word Documents

- **Chunk Size**: 500 characters
- **Best For**: Reports, articles, lengthy documents
- **Special Requirements**: NLTK punkt tokenizer

### 📋 PowerPoint Presentations

- **Chunk Size**: 200-300 characters
- **Best For**: Pitch decks, educational content
- **Advantage**: Extracts text boxes and slide components

### 📖 EPUB Files

- **Chunk Size**: 300-400 characters
- **Best For**: Ebooks, digital publications
- **Considerations**: Contains metadata and images

### 📑 PDF Files

- **Chunk Size**: 500-2200 characters (varies by content)
- **Best For**: Academic papers, contracts, manuals
- **Special Cases**: OCR needed for scanned PDFs

## 🎛️ Parameter Tuning Guide

### Chunk Size Optimization

- **Small chunks (200-500)**: Better for specific information retrieval
- **Large chunks (1000-2000)**: Better for maintaining context
- **Consider document type**: PowerPoint needs smaller chunks than PDFs

### Chunk Overlap Guidelines

- **Standard**: 10-20% of chunk size
- **Narrative content**: Higher overlap (20-25%)
- **Structured data**: Lower overlap (5-10%)

## 💡 Best Practices

### 🔧 Technical Considerations

- **API Key Security**: Always use environment variables
- **Cost Optimization**: Consider smaller embedding models for development
- **Performance**: Batch process large document collections
- **Memory Management**: Use appropriate chunk sizes to avoid token limits

### 📊 Quality Improvements

- **Preprocessing**: Clean documents before processing
- **Metadata Extraction**: Store document metadata separately for EPUB files
- **OCR Integration**: Use Tesseract for scanned PDFs
- **Multicolumn Handling**: Linearize text flow for academic papers

## 🔍 Example Queries

```python
# Load and process a document
db = process_document("reviews.xlsx")

# Ask specific questions
response = ask_question(db, "What are the worst reviews?")
print(response)

response = ask_question(db, "Analyze the feedback for improvements")
print(response)
```

## 📈 Performance Optimization

### Vector Store Options

- **FAISS**: Fast similarity search (used in examples)
- **Chroma**: Alternative with persistence
- **Pinecone**: Cloud-based solution for production

### Embedding Models

- **text-embedding-3-large**: Best performance (most expensive)
- **text-embedding-3-small**: Faster, cheaper alternative
- **Custom embeddings**: Cost-effective for specific domains

## 🚨 Troubleshooting

### Common Issues

- **Token Limits**: Reduce chunk size or overlap
- **Poor Retrieval**: Increase chunk overlap or adjust similarity threshold
- **Slow Processing**: Use smaller embedding models or batch processing
- **OCR Errors**: Preprocess scanned PDFs with better OCR tools

### Document-Specific Issues

- **Excel**: Use `table` mode for well-structured data
- **PowerPoint**: Increase chunk size for detailed slides
- **PDF**: Handle multicolumn layouts with preprocessing
- **EPUB**: Extract metadata separately for better indexing

---

**✨ Transform your documents into intelligent, searchable knowledge bases with RAG and LangChain!**
