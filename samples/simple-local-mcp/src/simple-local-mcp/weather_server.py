import logging

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    logger.info(f"Getting weather for {location}")
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="stdio")
