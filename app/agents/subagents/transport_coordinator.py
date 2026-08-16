"""
使用 Subagents 模式
"""

import asyncio
from langchain_openai import ChatOpenAI  #llm模型
from langchain.agents import create_agent  #创建主agent
from langchain.tools import tool 
from app.config import settings
from app.agents.subagents.flight_agent import create_flight_subagent
from app.agents.subagents.train_agent import create_train_subagent
from app.agents.subagents.driving_agent import create_driving_subagent
from app.mcp_core.client import get_mcp_client
from app.utils.logger import app_logger


# ============== 获取辅助工具 ==============

async def _get_auxiliary_tools():
    """获取辅助工具"""
    manager = await get_mcp_client()
    all_tools = await manager.get_tools()

    # 筛选辅助工具
    aux_tools = [
        tool for tool in all_tools
        if any(keyword in tool.name.lower() for keyword in [
            'getfutureweather', 'get-current-date',
            'maps_around_search'
        ])
    ]

    app_logger.info(f"🛠️ 辅助工具: {[t.name for t in aux_tools]}")
    return aux_tools


# ============== 创建交通规划主 Agent ==============

async def create_transport_coordinator():
    """
    创建交通规划协调器（主 Agent）

    整合：
    - 3个子Agent工具（航班、高铁、自驾）
    - MCP辅助工具（酒店、天气、日期、周边搜索）
    """

    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.llm_api_key,
        base_url=settings.base_url,
        temperature=0.3  # 查询类任务用低温度
    )

    # 异步创建子Agents
    # flight_subagent = await create_flight_subagent()
    train_subagent = await create_train_subagent()
    driving_subagent = await create_driving_subagent()

    # ============== 将 Subagents 包装为 Tools ==============

    # @tool("query_flights", description="查询航班信息。需要提供出发城市、目的地城市、出发日期。")
    # async def query_flights_tool(
    #     origin: str,
    #     destination: str,
    #     departure_date: str
    # ) -> str:
    #     """查询航班信息（调用航班 Subagent）"""

    #     app_logger.info(f"🔧 调用航班 Subagent: {origin} -> {destination}")

    #     result = await flight_subagent.ainvoke({
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": f"请查询从 {origin} 到 {destination} 的航班，出发日期是 {departure_date}。"
    #             }
    #         ]
    #     })

    #     return result["messages"][-1].content

    @tool("query_trains", description="查询高铁/火车信息。需要提供出发城市、目的地城市、出发日期。")
    async def query_trains_tool(
        origin: str,
        destination: str,
        departure_date: str
    ) -> str:
        """查询高铁信息（调用高铁 Subagent）"""

        app_logger.info(f"🔧 调用高铁 Subagent: {origin} -> {destination}")

        result = await train_subagent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"请查询从 {origin} 到 {destination} 的高铁，出发日期是 {departure_date}。"
                }
            ]
        })

        return result["messages"][-1].content

    @tool("plan_driving_route", description="规划自驾路线。需要提供出发地和目的地（地址或地名）。")
    async def plan_driving_route_tool(
        origin: str,
        destination: str
    ) -> str:
        """规划自驾路线（调用自驾 Subagent）"""

        app_logger.info(f"🔧 调用自驾 Subagent: {origin} -> {destination}")

        result = await driving_subagent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"请规划从 {origin} 到 {destination} 的自驾路线。"
                }
            ]
        })

        return result["messages"][-1].content

    # 获取辅助工具
    auxiliary_tools = await _get_auxiliary_tools()

    # 组合所有工具
    all_tools = [
        #query_flights_tool,
        query_trains_tool,
        plan_driving_route_tool,
    ] + auxiliary_tools

    coordinator = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt="""你是交通规划协调专家。

**可用工具**：

**交通查询（主要）**：
1. query_flights：查询航班信息（适合长途，速度快）(现在不能使用,不准调用这个工具)
2. query_trains：查询高铁信息（适合中短途，舒适便捷）
3. plan_driving_route：规划自驾路线（适合深度游，自由灵活）

**辅助工具（按需使用）**：
- getFutureWeatherByAirport：查询机场未来天气(现在不能使用,不准调用这个工具)
- get-current-date：获取今天日期（调用需要时间的工具前,获取实时时间）

**工作流程**：
1. 理解用户的交通需求（出发地、目的地、日期、人数）
2. 如果用户明确指定交通方式，直接调用对应工具
3. 如果用户未指定，根据距离推荐：
   * < 300km：推荐高铁
   * 300-1000km：推荐高铁或航班
   * > 1000km：推荐航班
4. 调用工具后，把工具的返回结果【原样】输出给用户
5. 可以主动询问用户偏好（时间优先还是价格优先）

**输出规则（最重要，必须遵守）**：
- query_trains、plan_driving_route 返回的内容已经是完整、格式化好的最终答案。
- 你必须【原样输出】工具返回的内容，严禁重新排版、重新总结、重新罗列车次/路线，严禁把工具结果再复述一遍。
- 最多在工具结果前面加一句简短的引导语（例如"为您查询到以下高铁方案："），正文直接粘贴工具返回内容，一字不改。
- 对同一个查询只调用一次工具，禁止为了"确认/补充"而重复调用同一工具。

**注意事项**：
- 用户说的今天和明天之类的词要以get-current-date获取的时间为标准
- 一定要调用工具获取实时信息，不要编造数据
- 如果查询失败，告知用户并提供替代方案
- 航班和高铁需要提供日期，自驾不需要
- 调用工具前，先调用get-current-date获取今天日期
"""
    )

    app_logger.info("✅ 交通规划协调器（主 Agent）创建完成")

    return coordinator


# if __name__ == "__main__":
#     async def main():
#         print("\n" + "=" * 50)
#         print("🚀 正在初始化交通规划协调器...")
#         print("=" * 50)

#         coordinator = await create_transport_coordinator()

#         test_query = "我想从北京去上海，明天出发，帮我推荐交通方式"

#         print(f"\n❓ 用户提问: {test_query}")
#         print("-" * 30)

#         response = await coordinator.ainvoke({
#             "messages": [{"role": "user", "content": test_query}]
#         })

#         print("-" * 30)
#         print("✅ 协调器回复:")
#         final_message = response["messages"][-1].content
#         print(final_message)

#         print("\n" + "=" * 50)
#         print("测试结束")
#         print("=" * 50)

#     asyncio.run(main())