import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

def load_emails_from_json(file_path):
    """Load emails from the extracted JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    emails = []
    if "emails" in data:
        for date_key, email_list in data["emails"].items():
            for email in email_list:
                emails.append(email)
    return emails

def format_email_for_embedding(email):
    """Format the email dictionary into a single string for better semantic chunking."""
    sender_name = email.get('sender', {}).get('name', '')
    sender_email = email.get('sender', {}).get('email', '')
    sender_info = f"{sender_name} <{sender_email}>" if sender_name else sender_email
    
    recipients = email.get('recipients', [])
    recipient_strs = []
    for r in recipients:
        r_name = r.get('name', '')
        r_email = r.get('email', '')
        r_info = f"{r_name} <{r_email}>" if r_name else r_email
        recipient_strs.append(r_info)
    
    recipients_info = ", ".join(recipient_strs)
    
    formatted_text = (
        f"Date: {email.get('time', 'Unknown')}\n"
        f"Subject: {email.get('subject', 'No Subject')}\n"
        f"From: {sender_info}\n"
        f"To: {recipients_info}\n"
        f"\n-- Body --\n{email.get('body', '')}"
    )
    return formatted_text

def main():
    json_path = "email_data.json"
    persist_dir = "chroma_db"
    embedding_model = "all-MiniLM-L6-v2"

    print(f"Loading emails from {json_path}...")
    raw_emails = load_emails_from_json(json_path)
    if not raw_emails:
        print("No emails found to process. Exiting.")
        return

    print(f"Formatting {len(raw_emails)} emails into documents...")
    documents = []
    for email in raw_emails:
        text = format_email_for_embedding(email)
        # Adding metadata to the document so we can filter during RAG if needed
        metadata = {
            "subject": email.get("subject", ""),
            "date": email.get("time", ""),
            "sender_email": email.get("sender", {}).get("email", "")
        }
        documents.append(Document(page_content=text, metadata=metadata))

    print(f"Initializing SemanticChunker with {embedding_model} embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Using SemanticChunker to split semantically based on embedding similarity
    text_splitter = SemanticChunker(embeddings)
    
    print("Chunking documents semantically (this might take a little while depending on the size)...")
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} semantic chunks from {len(documents)} emails.")

    print(f"Storing chunks in ChromaDB at {persist_dir}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print("Success! Embeddings stored inside local ChromaDB.")
    
    # Optional: Quick verification query
    print("\n--- Running a sample verification query ---")
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke("What are some important recent updates?")
    for doc in docs:
        print(f"Sample retrieved chunk: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()
