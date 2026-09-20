"""
Connects to mcp_server.py as an MCP client and calls both tools directly,
with no AI/LLM involved yet — just verifying the MCP server itself works
correctly in isolation.

Run with:
    python test_mcp.py
"""
import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # This describes HOW to launch the server: run "python mcp_server.py"
    # as a subprocess. The client will talk to it over stdin/stdout.
    server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # MCP handshake

            # List the tools this server exposes — confirms the server
            # started correctly and both tools registered properly.
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()

            # Call the weather tool directly
            print("Calling get_weather_forecast(city='Singapore', days=3)...")
            weather_result = await session.call_tool(
                "get_weather_forecast", arguments={"city": "Singapore", "days": 3}
            )
            print(json.dumps(json.loads(weather_result.content[0].text), indent=2))
            print()

            # Call the currency tool directly
            print("Calling convert_currency(amount=50000, from_currency='INR', to_currency='SGD')...")
            currency_result = await session.call_tool(
                "convert_currency",
                arguments={"amount": 50000, "from_currency": "INR", "to_currency": "SGD"},
            )
            print(json.dumps(json.loads(currency_result.content[0].text), indent=2))


if __name__ == "__main__":
    asyncio.run(main())