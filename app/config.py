"""
配置管理模块
使用 pydantic-settings 管理环境变量
读取 .env, 进行类型转换、参数校验
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import cache

# 获取当前文件的上级目录（即项目根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    """应用配置"""

    # ============== 应用基础配置 ==============
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    # ============== LLM 配置 ==============
    llm_api_key: str = Field(alias="LLM_API_KEY")
    model_name: str = Field(default="deepseek-v4-flash", alias="MODEL_NAME")
    base_url: str = Field(
        default="https://api.deepseek.com",
        alias="BASE_URL"
    )
    temperature: float = 0.7
    max_tokens: int = 8000

    rerank_model_api_key: str = Field(alias="RERANK_API_KEY")
    rerank_model_name: str = Field(default="qwen-turbo", alias="RERANK_MODEL")

    # ============== LangSmith 配置 ==============
    langsmith_api_key: str = Field(alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="travel-planner-dev", alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")
    langsmith_sampling_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        alias="LANGSMITH_TRACING_SAMPLING_RATE",
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGSMITH_ENDPOINT"
    )

    # ============== 数据库配置 ==============
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(alias="POSTGRES_DB")
    postgres_user: str = Field(alias="POSTGRES_USER")
    postgres_password: str = Field(alias="POSTGRES_PASSWORD")

    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")

    # ============== MCP 服务配置 ==============
    amap_api_key: str = Field(default="", alias="AMAP_API_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),     # 自动拼接路径，不管代码在哪运行都能找到
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"              # 忽略 .env 中多余的字段，防止报错
    )

    @property
    def database_url(self) -> str:
        """生成 PostgreSQL 连接字符串"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """生成 Redis 连接字符串"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@cache
def get_settings() -> Settings:
    """获取配置单例（缓存）"""
    return Settings()


# 全局配置对象
settings = get_settings()

# 测试
# if __name__ == '__main__':
#     print(settings.database_url)
#     print(settings.redis_url)
#     print(settings.qwen_model_name)
#     print(settings.qwen_base_url)
#     print(settings.qwen_temperature)
#     print(settings.qwen_max_tokens)
#     print(settings.langsmith_api_key)
