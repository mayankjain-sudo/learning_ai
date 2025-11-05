from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json

def load_and_split_pdfs(data_dir="./data", chunk_size=200, chunk_overlap=50):
    """
    Load all PDFs from the specified directory and split them into chunks.

    Args:
        data_dir (str): Directory containing PDF files.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of document chunks.
    """
    documents = []
    metadata = {}

    # Load metadata if available
    metadata_file = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    if os.path.exists(data_dir):
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            loader = PyPDFLoader(os.path.join(data_dir, pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf_file
                # Add metadata from JSON if available
                if pdf_file in metadata:
                    doc.metadata.update(metadata[pdf_file])
            documents.extend(docs)

    print(f"Number of documents: {len(documents)}")

    # Use smaller chunk size for better retrieval granularity
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")

    return chunks
