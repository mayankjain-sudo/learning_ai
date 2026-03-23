import sys
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_query.py \"Your question here\"")
        print("Example: python rag_query.py \"Did Grafana send any emails today?\"")
        sys.exit(1)
        
    query = sys.argv[1]
    
    persist_dir = "chroma_db"
    embedding_model = "all-MiniLM-L6-v2"
    llm_model = "llama3.2:latest"

    print("Initializing embeddings and connecting to local Chroma vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    print(f"Connecting to local Ollama LLM ({llm_model})...")
    llm = ChatOllama(model=llm_model, temperature=0.0)

    print("Constructing Semantic Retriever...")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    prompt_template = """
    You are an email assistant. Below are email documents retrieved from a personal inbox.
    Your task is to answer the user's question STRICTLY based on the email documents provided.
    
    STRICT RULES:
    1. Only use information from the provided email documents. Do NOT add any information beyond what is in the emails.
    2. If the user asks to show emails about a topic, list the emails found: their subject, sender, and a brief 1-sentence summary of the body.
    3. If no relevant emails are found in the context, say exactly: "I couldn't find any relevant emails for your query."
    4. Do NOT add questions, commentary, or analysis about why the email was sent.
    
    Retrieved Email Documents:
    {context}
    
    User Question: {question}
    
    Answer (summarize only what the emails say):"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    print("Executing RAG chain...")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(query)
    
    print("\n" + "="*50)
    print(f"QUESTION: {query}")
    print("="*50)
    print(f"ANSWER:\n{result}\n")
    print("="*50)
    
    print("SOURCES (Metadata Context):")
    source_docs = retriever.invoke(query)
    for i, doc in enumerate(source_docs, 1):
        subj = doc.metadata.get("subject", "Unknown")
        date = doc.metadata.get("date", "Unknown")
        print(f"  {i}. Subject: '{subj}' | Date: {date}")

if __name__ == "__main__":
    main()
