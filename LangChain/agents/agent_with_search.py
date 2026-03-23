import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables. Please check your .env file.")

print(f"TAVILY_API_KEY loaded: {TAVILY_API_KEY[:5]}...")  # Print first 5 chars only for security

# Set environment variable for Tavily
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Initialize search tool
search = TavilySearchResults(max_results=3)

# Initialize LLM
llm = ChatOllama(model="llama3.2")

# Define tools
tools = [search]

# Create memory checkpointer for conversation history
memory = MemorySaver()

# Create the agent with tools and memory
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)

# Agent executor function
def execute_agent(user_input: str, thread_id: str = "default-session"):
    """
    Execute the agent with a user query.
    
    Args:
        user_input: The user's question or command
        thread_id: Session identifier for conversation continuity
    
    Returns:
        The agent's response
    """
    # Properly formatted config for LangGraph
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    response = agent.invoke(
        {"messages": [("user", user_input)]},
        config
    )
    
    return response["messages"][-1].content

# Example usage
if __name__ == "__main__":
    try:
        # First query
        result = execute_agent("hi!")
        print(f"Response: {result}")
        
        # Follow-up query (uses conversation history)
        result2 = execute_agent("When is ICC Men's T20 World Cup 2024 scheduled?", thread_id="default-sessionclear")
        print(f"Response: {result2}")
        
        result3 = execute_agent("How many days before the first match start?", thread_id="default-sessionclear")
        print(f"Response3: {result3}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()




