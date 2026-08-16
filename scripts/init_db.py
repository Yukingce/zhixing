"""
数据库初始化脚本
"""
import asyncio
import sys
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from psycopg_pool import AsyncConnectionPool
from app.config import settings
from app.utils.logger import app_logger
from app.models.base import init_db


async def init_database():
    """初始化业务表 + pgvector 扩展"""

    db_url = settings.database_url
    app_logger.info(f"连接数据库: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")

    try:
        # 1. 初始化业务表（用户、会话、消息）
        app_logger.info("初始化业务表...")
        await init_db()
        app_logger.info("✅ 业务表创建成功")

        # 2. 启用 pgvector 扩展
        # 说明：Checkpointer / Store 的表已改由 app/core 中的管理器在启动时 setup() 创建，
        #       这里只保留 pgvector 扩展的启用（需要数据库用户有相应权限）。
        async with AsyncConnectionPool(conninfo=db_url, min_size=2, max_size=10) as pool:
            # 作用：让 PostgreSQL 支持 vector 向量类型，用于语义检索/RAG
            app_logger.info("启用 pgvector 扩展...")
            # pool.connection()：从连接池借出一个连接，用完自动归还
            async with pool.connection() as conn:
                # conn.cursor()：创建游标，用于执行 SQL
                async with conn.cursor() as cur:
                    # CREATE EXTENSION IF NOT EXISTS：幂等地启用 pgvector
                    # 注意：需要数据库用户有超级用户权限，否则会失败
                    await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    # DDL 必须显式 commit 才生效；
                    # 若在 async with 内不 commit，退出时会回滚，扩展不会被真正创建
                    await conn.commit()
            app_logger.info("✅ pgvector 扩展启用成功")

    except Exception as e:
        app_logger.error(f"❌ 数据库初始化失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(init_database())
