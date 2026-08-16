"""LangSmith 追踪配置与生命周期辅助函数。"""

import os
from typing import Any

from langchain_core.tracers.langchain import wait_for_all_tracers

from app.config import settings
from app.utils.logger import app_logger


def configure_langsmith() -> None:
    """把 Settings 中已校验的配置同步给读取环境变量的 LangSmith SDK。

    Docker Compose 会直接注入这些变量；这一步同时保证本地直接启动
    Uvicorn 时也能启用相同的自动追踪行为。
    """
    environment = {
        "LANGSMITH_TRACING": str(settings.langsmith_tracing).lower(),
        "LANGSMITH_API_KEY": settings.langsmith_api_key,
        "LANGSMITH_PROJECT": settings.langsmith_project,
        "LANGSMITH_ENDPOINT": settings.langsmith_endpoint,
        "LANGSMITH_TRACING_SAMPLING_RATE": str(
            settings.langsmith_sampling_rate
        ),
    }
    os.environ.update(environment)


def build_agent_trace_config(conversation_id: str) -> dict[str, Any]:
    """构造顶层 Agent 的 RunnableConfig。

    tags 和 metadata 会由 LangChain 传播给模型、工具和子 Agent，便于在
    LangSmith 中按环境、模型和会话检索整条调用链。
    """
    return {
        "run_name": "zhixing_travel_agent",
        "tags": [
            "agent:travel",
            f"env:{settings.app_env}",
        ],
        "metadata": {
            "environment": settings.app_env,
            "model": settings.model_name,
            "conversation_id": conversation_id,
        },
        "configurable": {
            "thread_id": conversation_id,
        },
    }


def log_langsmith_status() -> None:
    """记录追踪状态，不输出 API Key 等敏感配置。"""
    if settings.langsmith_tracing:
        app_logger.info(
            "LangSmith 追踪已启用，项目: {}，采样率: {:.0%}",
            settings.langsmith_project,
            settings.langsmith_sampling_rate,
        )
    else:
        app_logger.info("LangSmith 追踪未启用")


def flush_langsmith_traces() -> None:
    """等待 LangChain 的后台追踪任务完成并刷新上传缓冲区。

    LangSmith 上传失败不应阻止 FastAPI 正常关闭，因此这里只记录异常。
    """
    if not settings.langsmith_tracing:
        return

    try:
        wait_for_all_tracers()
        app_logger.info("LangSmith 待上传 Trace 已刷新")
    except Exception:
        app_logger.exception("LangSmith Trace 刷新失败")
