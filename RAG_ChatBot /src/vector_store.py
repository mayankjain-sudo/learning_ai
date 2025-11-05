from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks, embedding_model="nomic-embed-text", persist_directory=None):
    """
    Create a FAISS vector store from document chunks.

    Args:
        chunks (list): List of document chunks.
        embedding_model (str): Name of the embedding model.
        persist_directory (str): Directory to persist the vector store.

    Returns:
        FAISS: The FAISS vector store.
    """
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    if persist_directory:
        vector_store.save_local(persist_directory)
    return vector_store

def load_vector_store(persist_directory, embedding_model="nomic-embed-text"):
    """
    Load an existing FAISS vector store.

    Args:
        persist_directory (str): Directory where the vector store is persisted.
        embedding_model (str): Name of the embedding model.

    Returns:
        FAISS: The loaded FAISS vector store.
    """
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
    return vector_store
