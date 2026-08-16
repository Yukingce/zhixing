import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    client = MultiServerMCPClient({
        "12306-mcp": {
            "command": "npx.cmd" if os.name == "nt" else "npx",
            "args": ["-y", "12306-mcp@0.3.10"],
            "transport": "stdio",
            "env": os.environ.copy(),
        }
    })

    tools = await client.get_tools(server_name="12306-mcp")

    print(f"加载工具数量：{len(tools)}")
    for tool in tools:
        print(f"- {tool.name}")


asyncio.run(main())