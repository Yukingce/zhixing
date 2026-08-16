# 1. 选择运行环境
FROM python:3.12-slim

# 2. 设置容器内工作目录
WORKDIR /app

# 从 uv 官方镜像复制 uv；使用锁文件安装固定依赖
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

# 3. Python 运行设置
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy

# 4. 安装系统依赖
# curl 用于健康检查；Node.js >=18 用于本地 12306 MCP
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nodejs \
    npm \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，便于复用 Docker 构建缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# 7. 复制项目代码
# 再复制业务代码和数据
COPY app ./app
COPY scripts ./scripts
COPY data ./data

# 安装并编译 12306 MCP；其 package.json 要求 Node.js >=18
RUN cd app/mcp_core/12306-mcp \
    && npm ci \
    && npm run build

RUN mkdir -p /app/logs

# 8. 声明应用端口
EXPOSE 8000

# 9. 设置健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]