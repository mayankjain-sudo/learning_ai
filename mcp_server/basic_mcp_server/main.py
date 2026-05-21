from mcp.server.fastmcp import FastMCP, Context
import time
import logging

mcp = FastMCP("add_integers")

class MCPError(Exception):
    def __init__(self, code: int,message: str):
        super().__init__(f"[{code}] {message}")
        self.message = message
        self.code = code

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='mcp_server.log',
    filemode='a',
    force=True
)

logger = logging.getLogger(__name__)

@mcp.tool()
def add_integers(a: int, b: int) -> int:
    '''
    Add two integers together and return the result.
    Args:
        a (int): The first integer to add.
        b (int): The second integer to add.
    Returns:
        int: The sum of the two integers.
    '''
    logger.info(f"Adding integers: {a} + {b}")
    result = a + b
    logger.info(f"Result of addition: {result}")
    return result

@mcp.tool()
def divide(a: int, b: int) -> float:
    '''
    Divide two integers and return the result.
    Args:
        a (int): The numerator.
        b (int): The denominator.
    Returns:
        float: The result of the division.
    '''
    logger.info(f"Dividing integers: {a} / {b}")
    if b == 0:
        raise MCPError(code=400, message="Cannot divide by zero.")
    result =  a / b
    logger.info(f"Result of division: {result}")
    return result

@mcp.tool()
def long_process(steps: int):
    '''
    Simulate a long-running process by sleeping for a specified number of steps.
    Args:
        steps (int): The number of steps to simulate. Each step takes 1 second.
    '''
    for i in range(steps):
        print(f"Processing step... {i+1} of {steps}")
        time.sleep(0.1)
    return "Process completed."


if __name__ == "__main__":
    mcp.run(transport='stdio')
