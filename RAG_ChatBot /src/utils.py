from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()
    return {
        "vector_store_path": os.getenv("VECTOR_STORE_PATH"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "llm_model": os.getenv("LLM_MODEL"),
    }

def is_greeting(text):
    """Check if the input text is a simple greeting."""
    greetings = ['hi', 'hey', 'hello', 'greetings', 'good morning', 'good afternoon', 'good evening', 'hi there', 'hey there']
    return text.lower().strip() in greetings

def get_greeting_response():
    """Return a standard greeting response."""
    return "Hello! I am Maya, your AI assistant here to help you with questions about the documents. How can I assist you today?"
