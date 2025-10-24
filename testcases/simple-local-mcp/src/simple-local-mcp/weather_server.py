from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return f"The weather in {location} is sunny with a high of 75°F."

if __name__ == "__main__":
    mcp.run(transport="stdio")
