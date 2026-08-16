"""
测试 LLM 连接
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


def test_qwen_connection():
    """测试模型连接"""

    print("测试模型连接...")

    try:
        # 初始化模型
        model = ChatOpenAI(
            model="deepseek-v4-flash",
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("BASE_URL"),
            temperature=0.7
        )

        # 发送测试消息
        response = model.invoke([
            HumanMessage(content="你好，请用一句话介绍你自己。")
        ])

        print(f"✅ 连接成功！模型回复：\n{response.content}")

    except Exception as e:
        print(f"❌ 连接失败：{e}")
        raise


if __name__ == "__main__":
    test_qwen_connection()