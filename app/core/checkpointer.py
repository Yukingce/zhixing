"""
Checkpointer 管理器
===================

职责：为 LangGraph Agent 提供基于 PostgreSQL 的状态持久化（Checkpoint）。

它做了三件事：
1. 管理一个 PostgreSQL 异步连接池（psycopg_pool.AsyncConnectionPool）；
2. 创建并持有 LangGraph 官方的 AsyncPostgresSaver（检查点存储器）；
3. 用「单例 + 双重检查锁」保证整个应用只有一个连接池 / 一个 Checkpointer，
   并提供 FastAPI lifespan 生命周期钩子（启动初始化、关闭释放）。

为什么需要 Checkpointer？
-------------------------
LangGraph 每次执行 graph 时都会产生一个 state（对话状态、中间结果等）。
默认它只存在内存里，进程一重启就丢失。AsyncPostgresSaver 会把这个状态
快照（checkpoint）落库到 PostgreSQL，从而支持：
    - 断点续跑：任务执行到一半挂了，从上一个 checkpoint 恢复继续跑
    - 超时重试：超时 / 失败后从保存的状态重新发起
    - 并行分支协调：LangGraph 的 Send 分支各自保存 checkpoint，最后汇聚
    - human-in-the-loop：暂停图执行等人工确认，恢复时接着走

"""

import asyncio
from typing import Optional
from contextlib import asynccontextmanager  # 将一个异步生成器函数，转换成一个异步上下文管理器
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.config import settings
from app.utils.logger import app_logger


class CheckpointerManager:
    """Checkpointer 管理器（单例模式）

    说明：AsyncPostgresSaver 自身不负责管理连接，需要外部喂给它一个连接池
    （或连接字符串）。这里选择复用同一个 AsyncConnectionPool，而不是让它自己
    开连接，好处是应用里其它数据库操作也能共享这个池，避免连接数膨胀。
    """

    _instance: Optional['CheckpointerManager'] = None
    # 类级别的异步锁，用于保证多协程并发时单例只被初始化一次。
    # 注：Python 3.10+ 的 asyncio.Lock 创建时不再绑定事件循环（惰性绑定），所以定义成类属性是安全的。
    _lock = asyncio.Lock()

    def __init__(self):
        self.pool: Optional[AsyncConnectionPool] = None
        self.checkpointer: Optional[AsyncPostgresSaver] = None


    @classmethod
    async def get_instance(cls) -> 'CheckpointerManager':
        """获取单例实例（线程安全）

        为什么用「双重检查锁」而不是简单判空？
        因为 get_instance() 是 async 的，多个协程可能同时调用。若只判一次空，
        可能产生多个实例、多个连接池。所以分三步：
            1. 第一重检查（无锁）：已存在则直接返回，避开加锁开销（快速路径）
            2. 加锁：同一时刻只允许一个协程进入
            3. 第二重检查（有锁）：可能有两个协程同时通过第一重检查排队等锁，
               第二个拿到锁后必须再查一次，否则会重复初始化

        问题背景：get_instance() 是 async 的。也就是说，可能好几个协程同时喊「给我总机！」，如果谁喊都 new 一个，就会造出好几个连接池。
        解决思路是「只让第一个喊的人造，其他人等现成的」。但「判断有没有 + 造出来」这两步不是一瞬间完成的，中间可能被打断，所以要小心处理。代码用了三次判断：

        if cls._instance is None:          # ① 快速看一眼：已经有了？直接返回（省时间）
            async with cls._lock:          # ② 上锁：现在只允许一个人进来
                if cls._instance is None:  # ③ 再确认一次：排队期间别人可能已经造好了
                    cls._instance = cls()              # 造总机
                    await cls._instance.initialize()   # 让它开工（开池、建办事员）
        return cls._instance
        """

        # ━━━ 第一重检查（无锁，快速路径）━━━━━━━━━━━━━━━━
        if cls._instance is None:
            # ━━━ 获取锁（只有一个协程能进入）━━━━━━━━━━━
            async with cls._lock:
                # ━━━ 第二重检查（有锁，安全路径）━━━━━━━━━
                if cls._instance is None:
                    cls._instance = cls()        # 创建
                    await cls._instance.initialize() # 初始化
        return cls._instance

    async def initialize(self):
        """初始化连接池和 Checkpointer"""

        # ━━━ 第1步：防御性检查（幂等）━━━━━━━━━━━━━━━━━━━
        # get_instance() 理论上只初始化一次，但若有人手动调用 initialize()
        # 或异常路径重复进入，这里能跳过，避免重复开池。
        if self.checkpointer is not None:
            app_logger.warning("⚠️ Checkpointer 已初始化，跳过")
            return

        try:
            app_logger.info("初始化 PostgreSQL Checkpointer...")

            # 创建异步连接池
            self.pool = AsyncConnectionPool(
                conninfo=settings.database_url,  # PostgreSQL 连接字符串（见 config.py 的 database_url）
                min_size=2,  # 池中始终保持至少 2 个空闲连接，避免高并发下临时建连的开销
                max_size=20,  # 最多同时 20 个连接，超出则排队等待，防止打爆数据库
                # 如何确定 max_size 这个值？
                #   公式: max_size ≈ PostgreSQL 的 max_connections / 应用实例数
                #   例如: PG max_connections=100, 部署 4 个 Pod → max_size=25
                #   建议留一些余量给管理连接（如监控、迁移）
                timeout=30,  # 借连接最长等待 30 秒，超时抛异常（而非无限阻塞）
                # 注：默认 open=False（懒连接），下面显式 await pool.open() 触发预热，
                #     让 min_size 个连接提前建立好，降低首个请求的延迟。
                open=False
            )

            # 打开连接池（预热 min_size 个连接）
            await self.pool.open()

            # 创建 Checkpointer：把连接池交给 AsyncPostgresSaver 复用
            self.checkpointer = AsyncPostgresSaver(self.pool)

            # 创建 Checkpointer 所需的表（checkpoints / checkpoint_writes 等，幂等）
            await self.checkpointer.setup()

            app_logger.info("✅ Checkpointer 初始化完成")

        except Exception as e:
            app_logger.error(f"❌ Checkpointer 初始化失败: {e}")
            raise

    async def close(self):
        """关闭连接池

        在应用停机时调用，释放所有数据库连接。若池已建立则关闭并记日志；
        若从未初始化（self.pool 为 None）则静默跳过。
        """
        if self.pool:
            await self.pool.close()
            app_logger.info("Checkpointer 连接池已关闭")

    def get_checkpointer(self) -> AsyncPostgresSaver:
        """获取 Checkpointer 实例

        调用前必须先经过 get_instance()（内部已初始化）。若未初始化就调用，
        说明调用顺序有误，直接抛异常暴露问题，而不是返回一个半成品。
        """
        if self.checkpointer is None:
            raise RuntimeError("Checkpointer 未初始化，请先调用 initialize()")
        return self.checkpointer


# ============== 便捷函数 ==============

async def get_checkpointer() -> AsyncPostgresSaver:
    """
    获取全局 Checkpointer 实例

    用法：
        checkpointer = await get_checkpointer()
        graph = builder.compile(checkpointer=checkpointer)
    """
    manager = await CheckpointerManager.get_instance()
    return manager.get_checkpointer()


@asynccontextmanager
async def checkpointer_lifespan():
    """
    Checkpointer 生命周期管理器（用于 FastAPI lifespan）

    @asynccontextmanager 把一个异步生成器函数变成异步上下文管理器：
        - yield 之前的代码在进入 async with 时执行（初始化 / 获取实例）
        - finally 中的代码在退出 async with 时执行（保证释放连接池）

    用法：
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with checkpointer_lifespan():
                yield
    """
    manager = await CheckpointerManager.get_instance()
    try:
        yield manager.get_checkpointer()
    finally:
        await manager.close()
