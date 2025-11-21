import os
import json
from collections import defaultdict
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker

def process_pdfs_with_semantic_chunking(
    pdf_dir: str = "./../data",
    output_json_path: str = "semantic_chunks.json",
    chroma_persist_directory: str = "./chroma_db_semantic",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Loads all PDFs from a directory, chunks them semantically, stores chunks in ChromaDB,
    and writes chunks to a JSON file.

    Args:
        pdf_dir (str): The directory containing the PDF files.
        output_json_path (str): The path to save the JSON file with chunks.
        chroma_persist_directory (str): Directory to persist ChromaDB.
        embedding_model_name (str): Name of the HuggingFace embedding model to use.
    """
    # 1. Load PDFs
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"Directory not found at: {pdf_dir}")

    print(f"Loading PDFs from {pdf_dir}...")
    loader = PyPDFDirectoryLoader(pdf_dir)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDFs in the directory.")

    if not documents:
        print("No documents found.")
        return

    # Group pages by source file to reconstruct full documents
    docs_by_source = defaultdict(list)
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        docs_by_source[source].append(doc)

    full_texts = []
    metadatas = []
    
    for source, docs in docs_by_source.items():
        # Sort by page number if available, though PyPDFDirectoryLoader usually reads in order
        docs.sort(key=lambda x: x.metadata.get('page', 0))
        text = "\n\n".join([d.page_content for d in docs])
        full_texts.append(text)
        # Base metadata for the whole file
        metadatas.append({"source": source})

    # 2. Initialize embedding model
    print(f"Initializing embedding model: {embedding_model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # 3. Perform semantic chunking
    print("Performing semantic chunking...")
    text_splitter = SemanticChunker(embeddings)
    
    # create_documents takes a list of texts and optional list of metadatas
    chunks = text_splitter.create_documents(full_texts, metadatas=metadatas)
    print(f"Created {len(chunks)} semantic chunks from {len(full_texts)} documents.")

    # 4. Store chunks in ChromaDB
    print(f"Storing chunks in ChromaDB at {chroma_persist_directory}...")
    os.makedirs(chroma_persist_directory, exist_ok=True)
    
    collection_name = "semantic_chunks"
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_persist_directory,
        collection_name=collection_name
    )
    # vectordb.persist() # Not needed in newer Chroma versions
    print("Chunks successfully stored in ChromaDB.")

    # 5. Write chunks to a JSON file
    full_json_path = os.path.join(chroma_persist_directory, output_json_path)
    print(f"Writing chunks to JSON file at {full_json_path}...")
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunks_data.append({
            "chunk_id": i,
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        })

    with open(full_json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=4, ensure_ascii=False)
    print("Chunks successfully written to JSON file.")

if __name__ == "__main__":
    try:
        process_pdfs_with_semantic_chunking()
        print("\nProcessing complete!")
    except Exception as e:
        print(f"An error occurred: {e}")
    # from langchain_community.embeddings import OllamaEmbeddings
    # from langchain_community.vectorstores import Chroma
    # embeddings = OllamaEmbeddings(model="llama2")
    # db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    # print("\nQuery results:")
    # for doc in docs:
    #     print(doc.page_content[:100])
