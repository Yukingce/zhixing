"""
    全局中间件
    动态配置 Agent 行为

    修改 prompt 和 tools， 动态修改agent
"""

from jinja2 import Template
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from app.core.store import get_user_memory_service
from typing import Callable, Any
from app.core.state import TravelState
from app.utils.logger import app_logger


class StepConfigMiddleware(AgentMiddleware):
    """
        步骤配置中间件 - 根据 current_step 动态配置 Agent
    """

    def __init__(self, step_config: dict):
        """
        初始化中间件

        Args:
            step_config: 预加载的步骤配置字典
        """
        self._step_config = step_config


    async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """
        中间件钩子：在每次调用 LLM 之前拦截请求，按当前步骤动态配置 Agent。

        工作流程：
        1. 从 TravelState 中读取 current_step（当前处于哪一步）
        2. 校验步骤合法性、前置依赖字段是否齐全
        3. 用 state 里的字段填充该步骤 prompt 模板中的占位符
        4. 将配置好的 system_prompt 与 tools 注入本次请求
        5. 交给 handler 继续执行真正的 LLM 调用

        Args:
            request: 尚未发给 LLM 的请求，request.state 即 TravelState
            handler: 下游处理器，调用它才会真正发起 LLM 请求

        Returns:
            LLM 返回的 ModelResponse
        """
        # 1. 读取当前步骤（缺失时兜底为第一步「需求收集」）
        state: TravelState = request.state
        state_dict = dict(state) if hasattr(state, "items") else {}
        current_step = state.get("current_step", "requirement_collection")
        user_id = state.get("user_id")

        app_logger.info(f"📋 用户ID: {user_id}")
        app_logger.info(f"当前步骤: {current_step}")

        # 2. 步骤合法性校验：找不到对应配置直接报错，避免带病继续
        if current_step not in self._step_config:
            app_logger.error(f"❌ 未知步骤: {current_step}")
            raise ValueError(f"未知步骤: {current_step}")

        step_config = self._step_config[current_step]

        # 3. 验证前置依赖：确认进入该步骤所需的字段都已写入 state
        for required_field in step_config["requires"]:
            # requires 中列的是该步骤依赖的 TravelState 字段名
            if required_field not in state or state[required_field] is None:
                error_msg = f"步骤 {current_step} 需要完整状态: {required_field} 未设置"
                app_logger.error(f"❌ {error_msg}")
                app_logger.error(f"当前状态: {list(state.keys())}")
                raise ValueError(error_msg)

        # ========== 🔑 核心：注入长期记忆 ==========
        memory_prompt = ""
        if user_id:
            try:
                service = await get_user_memory_service()
                memory_prompt = await service.format_memory_for_prompt(user_id)
                if memory_prompt:
                    app_logger.info(f"💾 已加载用户长期记忆: {user_id}")
                else:
                    app_logger.info(f"📝 用户首次使用，暂无历史记忆: {user_id}")
            except Exception as e:
                app_logger.warning(f"⚠️ 加载长期记忆失败: {e}")


        # 4. 动态填充提示词模板
        try:
            # state_dict["user_memory"] = memory_prompt  # 将记忆注入到模板变量
            template = Template(step_config["prompt"])
            system_prompt = template.render(**state_dict) # 缺失变量更宽容
            
            # system_prompt = step_config["prompt"].format(**state)
            # 用 state 字段替换 prompt 里的 {xxx} 占位符
            # 注意：若模板写成 {user_requirement.departure_date} 这种「点号」形式，
            # format() 会对 dict 做属性访问并抛 AttributeError（而非 KeyError），
            # 下面的 except 无法兜住；嵌套字段应写成 {user_requirement[departure_date]}。
        except KeyError as e:
            # 占位符对应的字段缺失时，退回原始模板（不填充变量）
            app_logger.warning(f"⚠️ 提示词变量缺失: {e}，使用原始模板")
            system_prompt = step_config["prompt"]

        # 5. 注入配置：覆盖本次请求的 system_prompt 和可用工具集
        modified_request = request.override(
            system_prompt=system_prompt,
            tools=step_config["tools"] 
        )
        # 这是在中间件（wrap_model_call）里，对当前这一次模型调用临时覆盖工具列表。
        # 它不会修改 Agent 的永久配置，只影响本次请求中模型能"看到"哪些工具。

        app_logger.info(f"✅ 已注入步骤配置: {len(step_config['tools'])} 个工具")

        # 6. 把改造后的请求交给下游 handler，完成真正的 LLM 调用
        return await handler(modified_request) # 把当前 ModelRequest 继续交给后续调用链执行”的函数。


async def create_step_config_middleware() -> StepConfigMiddleware:
    """
    工厂函数：创建步骤配置中间件

    Returns:
        预加载配置的 StepConfigMiddleware 实例
    """
    from app.agents.handoffs.step_config import get_step_config

    step_config = await get_step_config()
    
    app_logger.info("✅ StepConfigMiddleware 创建完成")
    
    return StepConfigMiddleware(step_config)