"""
学习 Agent — FastAPI 接口
启动: python -m uvicorn api.main:app --reload --port 8000
"""
import sys
import os

# 把项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import run_agent, memory

app = FastAPI(title="学习 Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    context: str = ""


class ChatResponse(BaseModel):
    response: str
    intent: str
    memory_hits: list = []
    error: str = ""


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """对话接口"""
    try:
        result = run_agent(req.message, req.context)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory")
async def get_memory():
    """查看记忆状态"""
    return {
        "short_term_count": len(memory.short_term),
        "long_term_count": len(memory.long_term_texts),
        "recent": memory.get_recent(5),
    }


@app.delete("/api/memory")
async def clear_memory():
    """清除短期记忆"""
    memory.short_term = []
    return {"status": "短期记忆已清除"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "agent": "学习 Agent v0.1.0"}


# 挂载前端静态文件
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
