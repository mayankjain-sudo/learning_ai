import sys
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent

mcp = FastMCP("fixed-ollama-server")

OLLAMA_BASE_URL = 'http://localhost:11434'
DEFAULT_MODEL = 'llama3.2'

async def send_to_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send the prompt to the local Ollama instance and fetch the response."""
    try:
        # Using httpx.AsyncClient for non-blocking standard HTTP requests
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120.0
            )
            
            if response.status_code != 200:
                return f"HTTP Error {response.status_code}: {response.text}"
            
            result = response.json()
            return str(result.get("response", ""))
            
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.prompt(name="greeting", description="Creates a personalized greeting")
async def greeting() -> list[PromptMessage]:
    prompt = "Hello, Alice! Welcome to our MCP prompt demo."
    response = await send_to_ollama(prompt)
    
    return [
        PromptMessage(role="user", content=TextContent(type="text", text=prompt)),
        PromptMessage(role="assistant", content=TextContent(type="text", text=response))
    ]


@mcp.prompt(name="explain-concept", description="Explains complex concepts simply")
async def explain_concept() -> list[PromptMessage]:
    prompt = "Please explain quantum computing in simple terms that a beginner could understand. Use examples and analogies where appropriate."
    response = await send_to_ollama(prompt)
    
    return [
        PromptMessage(role="user", content=TextContent(type="text", text=prompt)),
        PromptMessage(role="assistant", content=TextContent(type="text", text=response))
    ]


@mcp.prompt(name="code-review", description="Reviews code and provides suggestions")
async def code_review() -> list[PromptMessage]:
    prompt = """Please review the following javascript code and provide suggestions for improvement:

```javascript
function hello() { console.log("hi") }
```

Please focus on:
- Code quality and best practices
- Performance optimizations  
- Security considerations
- Readability and maintainability"""
    
    response = await send_to_ollama(prompt)
    
    return [
        PromptMessage(role="user", content=TextContent(type="text", text=prompt)),
        PromptMessage(role="assistant", content=TextContent(type="text", text=response))
    ]

if __name__ == "__main__":
    print("✅ All prompts registered\n✅ Server ready", file=sys.stderr)
    mcp.run(transport='stdio')