import sys
import os

# Add project root to path for relative imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from src.utils import load_env, is_greeting, get_greeting_response
from src.data_loader import load_and_split_pdfs
from src.vector_store import create_vector_store, load_vector_store
import os
import json
from datetime import datetime

# Load environment variables
env = load_env()
vector_store_path = env["vector_store_path"]
embedding_model = env["embedding_model"] or "mxbai-embed-large"  # Upgraded for better semantic search
llm_model = env["llm_model"] or "llama3.2"

# Check if vector store already exists
if vector_store_path and os.path.exists(vector_store_path):
    # Load existing vector store
    vector_store = load_vector_store(persist_directory=vector_store_path, embedding_model=embedding_model)
else:
    # Load and process PDFs, create new vector store
    chunks = load_and_split_pdfs()
    vector_store = create_vector_store(chunks=chunks, embedding_model=embedding_model, persist_directory=vector_store_path)

# Base retriever with similarity score threshold for precise semantic search
base_retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 50, "score_threshold": 0.3})

# Use the base retriever without compression for now (compression requires additional setup)
retriever = base_retriever

def filter_by_metadata(docs, question):
    """
    Filter documents based on metadata to improve relevance.
    Prioritize policy documents for leave-related questions.
    """
    if "leave" in question.lower() or "policy" in question.lower():
        # Prioritize policy documents for leave/policy questions
        policy_docs = [doc for doc in docs if doc.metadata.get("type") == "policy"]
        if policy_docs:
            return policy_docs[:10]  # Return top 10 policy docs
    # For other questions, return all docs or prioritize handbook
    handbook_docs = [doc for doc in docs if doc.metadata.get("type") == "handbook"]
    if handbook_docs:
        return handbook_docs[:10]
    return docs[:10]  # Fallback to top 10

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

llm = OllamaLLM(model=llm_model)
template = """SYSTEM: You are a QnA bot name Maya.
            Be factual and concise in your answers.
            Respond to the following question: {question} only from
            the below context: {context}.
            If the context does not contain the information needed to answer the question, just say 'I am sorry I don't know',
            don't try to make up an answer.
            """
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def get_chat_history(input_dict):
    messages = []
    if "chat_history" in st.session_state:
        for q, a in st.session_state.chat_history:
            messages.append(HumanMessage(content=q))
            messages.append(AIMessage(content=a))
    return messages

def get_chain():
    def retrieve_and_filter(x):
        docs = retriever.invoke(x["question"])
        return filter_by_metadata(docs, x["question"])

    return (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(get_chat_history),
            context=RunnableLambda(retrieve_and_filter) | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )

chain = get_chain()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for chat history
st.sidebar.header("Chat History")
if st.sidebar.button("New Chat"):
    st.session_state.chat_history = []
    st.rerun()

for i, (q, a) in enumerate(st.session_state.chat_history):
    st.sidebar.write(f"**Q{i+1}:** {q[:50]}{'...' if len(q) > 50 else ''}")
    st.sidebar.write(f"**A{i+1}:** {a[:50]}{'...' if len(a) > 50 else ''}")
    st.sidebar.divider()

st.title("RAG Q&A Bot Maya")

# Display chat messages
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# Chat input
question = st.chat_input("Ask a question:")

if question:
    if is_greeting(question):
        answer = get_greeting_response()
    else:
        answer = chain.invoke({"question": question})
    st.session_state.chat_history.append((question, answer))
    st.rerun()
