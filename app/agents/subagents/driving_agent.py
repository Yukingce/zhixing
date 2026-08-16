"""
    自驾路线规划 Subagent
    调用高德地图 MCP 服务
"""
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from app.config import settings
from app.utils.logger import app_logger
from app.mcp_core.client import get_mcp_client



async def _get_amap_tools():
    """获取高德地图相关工具"""
    manager = await get_mcp_client()
    all_tools = await manager.get_tools()

    # 筛选地图工具
    amap_tools = [
        tool for tool in all_tools
        if any(keyword in tool.name.lower() for keyword in [
            'maps_direction_driving', 'maps_geo'
        ])
    ]

    app_logger.info(f"🚗 地图工具: {[t.name for t in amap_tools]}")
    return amap_tools

async def create_driving_subagent():
    """创建自驾路线规划 Subagent"""

    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.llm_api_key,
        base_url=settings.base_url,
        temperature=0.3  # 查询类任务用低温度
    )

    # 异步获取工具
    amap_tools = await _get_amap_tools()

    agent = create_agent(
        model=llm,
        tools=amap_tools,
        system_prompt="""你是自驾路线规划专家，负责处理驾车路线查询、距离时长估算及费用预估。可以使用以下工具：

**可用工具**：
1.  **地理编码**：
    - `maps_geo`: 将【地址或地名】转换为经纬度坐标（地理编码）。
      返回格式：经度,纬度（如 "116.481499,39.990475"）

2.  **路线规划**：
    - `maps_direction_driving`: 规划【驾车路线】。
      参数：起点经纬度、终点经纬度（必须是坐标格式，不能是地址）
      返回：总距离、预计时长、路线详情、过路费等信息

**查询流程**：
1.  **坐标转换**：
    - 严禁直接将地址传给 `maps_direction_driving`。
    - 必须先对出发地和目的地分别调用 `maps_geo` 获取经纬度。
    - 如果用户提供的是"北京"、"上海博物馆"等地名，都需要先转换为坐标。

2.  **路线规划**：
    - 获取到两个地点的经纬度后，调用 `maps_direction_driving` 规划路线。
    - 经纬度格式必须是：经度,纬度（逗号分隔，无空格）

3.  **错误处理**：
    - 如果 `maps_geo` 无法解析某个地址，请要求用户提供更详细或准确的地址。
    - 如果路线规划失败，请检查坐标格式是否正确。

**输出格式**：
请以结构化清晰的方式回答，包含：

🚗 **自驾路线方案**

**基本信息**：
- 起点：{起点地址}
- 终点：{终点地址}
- 总距离：{XX 公里}
- 预计时长：{XX 小时 XX 分钟}

**路线详情**：
{主要路段、道路名称、关键转弯等}

**费用估算**：
- 过路费：约 XX 元
- 油费：约 XX 元（按 7L/100km，油价 8 元/L 计算）
- 总费用：约 XX 元

**注意事项**：
{如拥堵提示、路况建议等}
"""
    )

    app_logger.info("✅ 自驾 Subagent 创建完成")
    return agent


