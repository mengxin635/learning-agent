"""
DeepSeek API 调用封装
优先级：项目 .env > ~/.hermes/.env（复用 Hermes 已配置的 Key）
"""
import json
import os
from dotenv import load_dotenv

# 先加载项目 .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
# 再加载 Hermes 全局 .env 作为回退
hermes_env = os.path.expanduser("~/.hermes/.env")
if os.path.exists(hermes_env):
    load_dotenv(hermes_env, override=False)  # override=False: 项目 .env 优先

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")


async def chat(messages: list, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """调用 DeepSeek Chat API，返回文本回复"""
    import httpx

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-api-key-here":
        return "[错误] 请先设置 .env 中的 DEEPSEEK_API_KEY"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        data = resp.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return f"[API 错误] {data}"


def chat_sync(messages: list, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """同步版，用于非 async 节点"""
    import httpx

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-api-key-here":
        return "[错误] 请先设置 .env 中的 DEEPSEEK_API_KEY"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = httpx.post(
        f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    data = resp.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    return f"[API 错误] {data}"
