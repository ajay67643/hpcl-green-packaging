import os
import shutil
from pathlib import Path
from typing import List

# --- LangChain Imports ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- Configuration ---
DATA_PATH = "Data"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "embeddinggemma:latest" # Model for efficient vectorization

# ----------------- Step 1: Data Preparation -----------------

def load_documents():
    """
    Loads all .txt documents from the DATA_PATH folder.
    Returns: List of LangChain documents.
    """
    # Check if data directory exists and contains files
    if not os.path.exists(DATA_PATH) or not any(Path(DATA_PATH).glob("*.txt")):
        print(f"Error: The '{DATA_PATH}' folder must exist and contain .txt files.")
        print("Please create the folder and add your data files before running the RAG system.")
        return []

    print(f"Loading documents from {DATA_PATH}...")
    # Use TextLoader with UTF-8 encoding to handle special characters
    def create_text_loader(file_path):
        return TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
    
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=create_text_loader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")
    return documents

def split_text(documents):
    """
    Splits documents into smaller, manageable chunks (tokenization proxy).
    This process is crucial for effective retrieval (RAG).
    Returns: List of LangChain document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=70,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

# ----------------- Step 2: Vector Database (Chroma) Setup -----------------

def build_database():
    """
    Builds the Chroma vector store, which performs the embedding/vectorization, 
    from the documents and saves it to disk.
    Returns: Chroma database instance, or None if no documents were loaded.
    """
    documents = load_documents()
    if not documents:
        return None

    # Clear existing data
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Removed old database at {CHROMA_PATH}")

    chunks = split_text(documents)

    # Initialize Ollama Embeddings for the vector database (Vectorization)
    print(f"Initializing Ollama Embeddings (model: {EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Create the vector store and persist it
    print(f"Creating Chroma database at {CHROMA_PATH}...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Database built successfully.")

    return db

def get_retriever():
    """
    Loads the existing Chroma database or builds it if necessary, 
    and returns a LangChain Retriever object for RAG.
    Returns: LangChain Retriever, or None.
    """
    if not os.path.exists(CHROMA_PATH):
        print(f"Database not found at {CHROMA_PATH}. Building new database...")
        db = build_database()
        if not db:
            return None
    else:
        print(f"Loading existing database from {CHROMA_PATH}...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # The retriever is configured to fetch the top 3 most relevant documents (k=3)
    return db.as_retriever(search_kwargs={"k": 2})

# If this file is run directly, it will build the database
if __name__ == "__main__":
    build_database()
