import logging

from mcp.server.mcpserver import MCPServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = MCPServer("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    result = a + b
    logger.info(f"Adding {a} and {b}: {result}")
    return result

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    result = a * b
    logger.info(f"Multiplying {a} and {b}: {result}")
    return result

if __name__ == "__main__":
    mcp.run(transport="stdio")
