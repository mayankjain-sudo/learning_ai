import sys
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

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

    print("Constructing Intent-Based Self-Query Retriever...")
    metadata_field_info = [
        AttributeInfo(
            name="subject",
            description="The subject line of the email",
            type="string",
        ),
        AttributeInfo(
            name="sender_email",
            description="The email address of the person who sent the email",
            type="string",
        ),
        AttributeInfo(
            name="date",
            description="The timestamp or date the email was received",
            type="string",
        ),
    ]
    document_content_description = "The body and content of an email message"
    
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        search_kwargs={"k": 4}
    )

    prompt_template = """
    You are an intelligent virtual assistant analyzing a user's extracted emails.
    
    Use the following pieces of retrieved email context to answer the user's question accurately.
    
    CRITICAL RULES:
    1. Base your answer ONLY on the provided context.
    2. If the context does not contain the information needed to answer the question, firmly reply "I don't know the answer based on the provided email context." Do not make up internal details, dates, or senders.
    3. Keep your answer concise and direct (maximum 4 sentences).
    
    Context: {context}
    
    Question: {question}
    """
    
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
