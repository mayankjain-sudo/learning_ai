import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration Settings ---
st.set_page_config(page_title="Email RAG Assistant", page_icon="📧", layout="centered")

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:latest"

@st.cache_resource
def load_rag_chain():
    """Initializes the vector store, embeddings, and Langchain pipeline."""
    # 1. Initialize embeddings and DB
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    
    # 2. Initialize LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=0.0)

    # 3. Use a reliable semantic retriever (k=6 for broader recall)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    prompt_template = """
    You are an email assistant. Below are email documents retrieved from a personal inbox.
    Your task is to answer the user's question STRICTLY based on the email documents provided.
    
    STRICT RULES:
    1. Only use information from the provided email documents. Do NOT add any information, context, or analysis beyond what is explicitly in the emails.
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
        
    # 5. Assemble the Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

st.title("📧 Email RAG Assistant")
st.markdown("Ask questions about your extracted emails! The assistant uses **Intent-Based Query Routing** to automatically filter by dates and senders, preventing hallucinations.")

# Initialize the RAG resources
with st.spinner("Connecting to Vector Database and initializing Intent Router..."):
    try:
        rag_chain, retriever = load_rag_chain()
    except Exception as e:
        st.error(f"Failed to initialize the RAG engine. Is your local ChromaDB populated?\nError: {e}")
        st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Retrieved Sources"):
                    for i, doc in enumerate(message["sources"], 1):
                        subj = doc.metadata.get("subject", "Unknown")
                        date = doc.metadata.get("date", "Unknown")
                        sender = doc.metadata.get("sender_email", "Unknown")
                        st.markdown(f"**{i}. {subj}**")
                        st.caption(f"📅 Date: {date} | 👤 From: {sender}")
                        st.text(doc.page_content[:200] + "...")

# Accept user input
if prompt := st.chat_input("Ask a question about your emails (e.g., 'Did I get emails from Grafana?')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Analyzing intent and searching emails..."):
            response = rag_chain.invoke(prompt)
            source_docs = retriever.invoke(prompt)
            
            message_placeholder.markdown(response)
            
            with st.expander("View Retrieved Sources"):
                for i, doc in enumerate(source_docs, 1):
                    subj = doc.metadata.get("subject", "Unknown")
                    date = doc.metadata.get("date", "Unknown")
                    sender = doc.metadata.get("sender_email", "Unknown")
                    st.markdown(f"**{i}. {subj}**")
                    st.caption(f"📅 Date: {date} | 👤 From: {sender}")
                    st.text(doc.page_content[:200] + "...")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": source_docs
        })
