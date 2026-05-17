"""
DeepSeek API 调用封装 — 多模型路由

模型策略：
  pro:  deepseek-v4-pro  — 思考/架构/辅导/规划（需要深度推理）
  flash: deepseek-v4-flash — 出题/闲聊/代码生成（需要速度）

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
    load_dotenv(hermes_env, override=False)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 模型配置
MODEL_PRO = os.getenv("MODEL_PRO", "deepseek-v4-pro")      # 思考/架构/辅导
MODEL_FLASH = os.getenv("MODEL_FLASH", "deepseek-v4-flash")  # 生成/出题/闲聊
MODEL_FALLBACK = os.getenv("MODEL_NAME", "deepseek-chat")    # 兼容旧配置


def _get_model(kind: str = None) -> str:
    """根据类型选择模型"""
    if kind == "pro":
        return MODEL_PRO
    elif kind == "flash":
        return MODEL_FLASH
    return MODEL_FLASH  # 默认用 flash（快且便宜）


def _call_api(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """底层 API 调用"""
    import httpx

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-api-key-here":
        return "[错误] 请先设置 .env 中的 DEEPSEEK_API_KEY"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = httpx.post(
        f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    data = resp.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    return f"[API 错误] {data}"


async def chat(messages: list, temperature: float = 0.7, max_tokens: int = 2048,
               model_kind: str = "flash") -> str:
    """异步调用，支持模型选择"""
    model = _get_model(model_kind)
    return _call_api(messages, model, temperature, max_tokens)


def chat_sync(messages: list, temperature: float = 0.7, max_tokens: int = 2048,
              model_kind: str = "flash") -> str:
    """同步版，用于非 async 节点"""

    import httpx

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-api-key-here":
        return "[错误] 请先设置 .env 中的 DEEPSEEK_API_KEY"

    model = _get_model(model_kind)
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = httpx.post(
        f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    data = resp.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    return f"[API 错误] {data}"
