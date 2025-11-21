import os
import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings # Updated import
from langchain_community.vectorstores import Chroma

def load_and_chunk_pdfs_to_chroma(data_dir: str = "./../data", persist_directory: str = "chroma_db", json_output_path: str = "chunks.json"):
    """
    Loads all PDF files from the specified directory, chunks them,
    stores the chunks in a Chroma vector database, and saves them to a JSON file.

    Args:
        data_dir (str): The directory containing the PDF files.
        persist_directory (str): The directory to persist the Chroma database.
        json_output_path (str): The path to save the chunks as a JSON file.
    """
    # 1. Load PDFs from the data directory
    print(f"Loading PDFs from {data_dir}...")
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages.")

    if not documents:
        print("No PDF documents found to process.")
        return

    # 2. Chunk the documents
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Save chunks to JSON
    full_json_path = os.path.join(persist_directory, json_output_path)
    
    # Ensure persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    print(f"Saving chunks to JSON at {full_json_path}...")
    chunks_data = [
        {
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=4, ensure_ascii=False)
    print("Chunks saved to JSON successfully.")

    # 4. Initialize embeddings and Chroma DB
    print("Initializing embeddings and ChromaDB...")
    # Ensure Ollama is running
    embeddings = OllamaEmbeddings(model="llama3.2") 

    # Create Chroma vector store and add documents
    # Chroma automatically persists in newer versions
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Chunks stored successfully in Chroma DB at {persist_directory}")

if __name__ == "__main__":
    # Create a minimal valid PDF for demonstration
    
    load_and_chunk_pdfs_to_chroma()
    # You can then load the DB later:
    # from langchain_community.embeddings import OllamaEmbeddings
    # from langchain_community.vectorstores import Chroma
    # embeddings = OllamaEmbeddings(model="llama2")
    # db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    # query = "What is this document about?"
    # docs = db.similarity_search(query)
    # print("\nQuery results:")
    # for doc in docs:
    #     print(doc.page_content[:100])
