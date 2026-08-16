"""
Handoffs 主 Agent
一个 Agent + 中间件实现整个旅行规划流程
"""
from app.tools.mcp_tools import get_all_mcp_tools
from app.tools.router_query import query_destination_info
from app.tools.transport_query import query_transport_options
from langchain.agents import create_agent
from app.config import settings
from app.core.state import TravelState
from app.core.checkpointer import get_checkpointer
from langchain_openai import ChatOpenAI
from app.core.middleware import create_step_config_middleware
from app.tools.state_transition import (
    record_requirement_tool,
    select_destination_tool,
    select_transport_tool,
    select_accommodation_tool,
    select_food_tool,
    generate_itinerary_tool,
    summarize_budget_tool,
    generate_order_tool,
    ALL_ROLLBACK_TOOLS
)
from app.utils.logger import app_logger
from langgraph.checkpoint.memory import MemorySaver

from app.tools.rag_tools import get_rag_tools  # 导入 RAG 工具
from app.tools.memory_tools import MEMORY_TOOLS  # 导入记忆工具
# 获取所有 RAG 工具
rag_tools = get_rag_tools()


# ============== 初始化 LLM ==============

def get_llm():
    """获取配置好的模型"""
    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.base_url,
        api_key=settings.llm_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=60,
        streaming=True
    ).with_config({"tags": ["travel_agent_llm"]})


# ============== 创建 Agent ==============

async def create_travel_agent():
    """
    创建 Handoffs 旅行规划 Agent

    返回：
        编译好的 Agent（可直接调用）
    """

    app_logger.info("创建 Travel Agent...")

    llm = get_llm()

    all_mcp_tools = await get_all_mcp_tools()

    # 异步创建中间件（预加载配置）
    step_config_middleware = await create_step_config_middleware()

    all_tools = [
        record_requirement_tool,
        select_destination_tool,
        select_transport_tool,
        select_accommodation_tool,
        select_food_tool,
        generate_itinerary_tool,
        summarize_budget_tool,
        generate_order_tool,
        *ALL_ROLLBACK_TOOLS,
        query_destination_info,
        query_transport_options,
        *all_mcp_tools,
        *rag_tools,
        *MEMORY_TOOLS
    ]

    checkpointer = await get_checkpointer()

    agent = create_agent(
        model=llm,
        tools=all_tools,
        # 这是 Agent 创建时注册的完整工具池，是 Agent 在整个生命周期内"理论上可以调用"的所有工具。
        # 它会被持久化保存在 Agent 的配置中，作为默认的工具列表。
        
        state_schema=TravelState,
        middleware=[step_config_middleware],  # 使用预加载的中间件
        checkpointer=checkpointer,
    )
    
    app_logger.info("✅ Travel Agent 创建完成")

    return agent

