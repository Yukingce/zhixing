"""
流式对话 API（SSE）
"""
import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse #流式返回
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from langchain_core.messages import HumanMessage, AIMessage
from app.models.base import get_db
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.message import MessageCreate
from app.api.dependencies import get_current_user
from app.agents.handoffs.travel_agent import create_travel_agent
from app.observability.langsmith import build_agent_trace_config
from app.utils.logger import app_logger

router = APIRouter(prefix="/chat", tags=["对话"])

HEARTBEAT_INTERVAL_SECONDS = 15


async def save_message(
        db: AsyncSession,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict = None
) -> Message:
    """保存消息到数据库"""

    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        extra_info=metadata or {}
    )

    db.add(message)
    await db.commit()
    await db.refresh(message)

    return message


def sse(data: dict) -> str:
    """
    SSE 标准 data 帧
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def generate_sse_stream(
        conversation_id: str,
        user_message: str,
        db: AsyncSession,
        user: User
):
    assistant_message = ""
    agent_task = None

    try:
        # 1. 保存用户消息
        await save_message(db, conversation_id, "user", user_message)

        # 2. 创建 agent
        agent = await create_travel_agent()

        # 3. 关键修复：输入必须是字典格式！
        # LangGraph StateGraph 期望输入是 state 的部分更新
        input_data = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": str(user.id),
        }

        # 4. 完整执行顶层 Agent，不再转发嵌套 Agent 的模型 token。
        # 等待期间发送不含 content 的心跳，防止长耗时查询导致 SSE 连接空闲超时。
        agent_task = asyncio.create_task(
            agent.ainvoke(
                input_data,
                config=build_agent_trace_config(conversation_id),
            )
        )

        while not agent_task.done():
            done, _ = await asyncio.wait(
                {agent_task},
                timeout=HEARTBEAT_INTERVAL_SECONDS,
            )
            if not done:
                yield sse({"type": "heartbeat"})

        result = await agent_task
        messages = result.get("messages", [])
        final_message = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, AIMessage)
                and message.content
                and not getattr(message, "tool_calls", None)
            ),
            None,
        )

        if final_message is None:
            raise RuntimeError("Agent 未返回有效的最终回复")

        if not isinstance(final_message.content, str):
            raise RuntimeError("Agent 最终回复不是文本内容")

        assistant_message = final_message.content
        yield sse({
            "type": "token",
            "content": assistant_message,
        })

        # 5. 保存 AI 回复
        if assistant_message.strip():
            await save_message(
                db,
                conversation_id,
                "assistant",
                assistant_message,
            )

        yield sse({"type": "done"})

    except Exception as e:
        app_logger.exception("❌ SSE 流式对话错误")
        yield sse({
            "type": "error",
            "message": str(e),
        })
    finally:
        if agent_task is not None and not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass



@router.post("/stream/{conversation_id}")
async def stream_chat(
        conversation_id: str,
        data: MessageCreate,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """
    流式对话（SSE）

    Returns:
        StreamingResponse: SSE 流式响应
    """

    # 验证会话归属
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == user.id)
    )

    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在"
        )

    # 返回 SSE 流
    return StreamingResponse(
        generate_sse_stream(conversation_id, data.content, db, user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 Nginx 缓冲
        }
    )


@router.get("/history/{conversation_id}")
async def get_chat_history(
        conversation_id: str,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """获取会话历史消息"""

    # 验证会话归属
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == user.id)
    )

    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在"
        )

    # 查询消息
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )

    messages = result.scalars().all()

    return {
        "conversation": conversation.to_dict(),
        "messages": [m.to_dict() for m in messages]
    }
