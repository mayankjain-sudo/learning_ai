
import sys
import os

# Add project root to path for relative imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.utils import load_env, is_greeting, get_greeting_response
from src.data_loader import load_and_split_pdfs
from src.vector_store import create_vector_store, load_vector_store
import os

# Load environment variables
env = load_env()
vector_store_path = env["vector_store_path"]
embedding_model = env["embedding_model"] or "nomic-embed-text"
llm_model = env["llm_model"] or "llama3.2"

#To load the data form webpage

URL="https://ollama.com/library/nomic-embed-text"
#loader = WebBaseLoader(URL)
#To load the data from PDF for chat boat

# Check if vector store already exists
if vector_store_path and os.path.exists(vector_store_path) and os.listdir(vector_store_path):
    # Load existing vector store
    vector_store = load_vector_store(persist_directory=vector_store_path, embedding_model=embedding_model)
else:
    # Load and process PDFs, create new vector store
    chunks = load_and_split_pdfs()
    vector_store = create_vector_store(chunks=chunks, embedding_model=embedding_model, persist_directory=vector_store_path)

retriever = vector_store.as_retriever(search_kwargs={"k": 10, "filter": {"source": "LEAVE POLICY-G7CR.pdf"}})
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])
llm = OllamaLLM(model=llm_model or "llama3.2")
template="""SYSTEM: You are a QnA bot name Maya.
            Be factual and concise in your answers.
            Respond to the following question: {question} only from 
            the below context: {context}.
            If you don't know the answer, just say that you don't know,
            don't try to make up an answer.
            """
prompt = PromptTemplate.from_template(template)
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Test function for greeting detection
def test_greeting(question):
    if is_greeting(question):
        return get_greeting_response()
    else:
        return chain.invoke(question)

# Example usage
if __name__ == "__main__":
    # Test greeting
    print("Greeting test:", test_greeting("hi"))
    print("Question test:", test_greeting("How many leaves available during probation ?"))
