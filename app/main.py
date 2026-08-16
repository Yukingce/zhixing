"""
FastAPI 应用入口
集成 Checkpointer 和 Store 的生命周期管理
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.utils.logger import app_logger
from app.api.v1 import conversation, chat, users

import asyncio
import sys

# Windows 下 psycopg 的异步模式不支持默认的 ProactorEventLoop，
# 必须在事件循环创建前切换为 SelectorEventLoop（否则启动时连接池报错）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    import asyncio

    # 打印当前事件循环信息（调试用）
    loop = asyncio.get_running_loop()
    app_logger.info(f"FastAPI 使用的事件循环: {type(loop).__name__}")

    from app.core.checkpointer import checkpointer_lifespan
    from app.mcp_core.client import MCPClientManager
    from app.core.store import store_lifespan

    app_logger.info("🚀 启动应用...")

    async with checkpointer_lifespan():
        app_logger.info("✅ Checkpointer 已就绪")

        # 2. 启动 Store (在这里加入)
        async with store_lifespan():
            app_logger.info("✅ Store 已就绪")

            # 3. 初始化 MCP
            mcp = await MCPClientManager.get_instance()
            app_logger.info("✅ MCP 服务初始化成功")

            yield

            # --- 关闭流程 (自动按相反顺序执行 context manager 的 exit，这里手动关闭 MCP) ---

            await mcp.close()
            app_logger.info("MCP 服务已关闭")

        # store_lifespan 的 exit 会在这里自动执行

    # checkpointer_lifespan 的 exit 会在这里自动执行

    app_logger.info("应用已关闭")



# 创建 FastAPI 应用
app = FastAPI(
    title="LangGraph 旅行规划系统",
    description="企业级多 Agent 旅行规划服务",
    version="1.0.0",
    lifespan=lifespan  # ← 将生命周期管理器绑定到 FastAPI
)

# CORS 配置（允许前端跨域）：CORS (跨源资源共享) 是浏览器的安全机制。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应指定具体域名
    #意味着任何网站都可以调你的 API。
    #上线前必须改成 ["https://your-frontend-domain.com"]，否则会有严重的安全风险 。
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router, prefix="/api/v1")
app.include_router(conversation.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "LangGraph Travel Planner",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug
    )