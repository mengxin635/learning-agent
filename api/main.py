"""
学习 Agent — FastAPI 接口
启动: python -m uvicorn api.main:app --reload --port 8000

v1.2: 新增记忆系统接口（知识图谱、复习计划、学习档案）
"""
import sys
import os

# 把项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import run_agent, memory
from agent.rag import kb, KB_DIR
import shutil

app = FastAPI(title="学习 Agent", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 请求/响应模型 ==========

class ChatRequest(BaseModel):
    message: str
    context: str = ""


class ChatResponse(BaseModel):
    response: str
    intent: str
    memory_hits: list = []
    error: str = ""


class RecordLearningRequest(BaseModel):
    topic: str
    quality: float = 0.5
    related: list = []


# ========== 对话接口 ==========

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """对话接口"""
    try:
        result = run_agent(req.message, req.context)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok", "agent": "学习 Agent v1.2.0", "memory": memory.get_status()}


# ========== 记忆系统接口 ==========

@app.get("/api/memory")
async def get_memory():
    """获取完整记忆状态：工作记忆 + 长期记忆 + 知识图谱 + 用户档案"""
    return memory.get_status()


@app.get("/api/memory/status")
async def memory_status():
    """记忆状态摘要"""
    status = memory.get_status()
    return {
        "short_term": status["short_term"],
        "long_term": status["long_term"],
        "topics": status["knowledge_graph"]["total_topics"],
        "mastered": status["knowledge_graph"]["mastered"],
        "learning": status["knowledge_graph"]["learning"],
        "due_review": status["knowledge_graph"]["due_review"],
    }


@app.delete("/api/memory")
async def clear_memory():
    """清空短期对话记忆（保留知识图谱）"""
    memory.short_term = []
    return {"status": "ok", "message": "短期记忆已清空"}


# ========== 知识图谱接口 ==========

@app.get("/api/knowledge")
async def get_knowledge():
    """获取知识图谱总览"""
    return memory.get_knowledge_summary()


@app.get("/api/knowledge/topics")
async def get_all_topics():
    """获取所有已学主题及掌握度"""
    return {"topics": memory.kg.get_all_topics()}


@app.post("/api/knowledge/record")
async def record_knowledge(req: RecordLearningRequest):
    """手动记录学习（通常自动记录，此接口用于手动补充）"""
    memory.kg.record_learning(req.topic, req.quality, req.related if req.related else None)
    return {"status": "ok", "topic": req.topic}


@app.get("/api/knowledge/weak")
async def get_weak_topics(limit: int = 5):
    """获取掌握度最低的主题"""
    return {"weak_topics": memory.get_weak_topics(limit)}


@app.get("/api/knowledge/review-plan")
async def get_review_plan(limit: int = 5):
    """获取今日复习计划"""
    return {"review_plan": memory.get_review_plan(limit)}


# ========== 知识库接口 ==========

@app.post("/api/kb/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文档到知识库"""
    allowed = {".pdf", ".md", ".txt", ".py", ".js", ".html", ".json"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"不支持的格式: {ext}，支持: {', '.join(allowed)}")

    tmp_path = os.path.join(KB_DIR, f"_tmp_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    count = kb.ingest_file(tmp_path)
    for chunk in kb.chunks:
        if chunk["source"] == os.path.basename(tmp_path):
            chunk["source"] = file.filename or "uploaded"
    kb._save()
    os.remove(tmp_path)

    return {"status": "ok", "filename": file.filename, "chunks_added": count}


@app.post("/api/kb/ingest")
async def ingest_text(text: str = Form(...), source: str = Form("manual")):
    """直接摄入文本"""
    count = kb.ingest_text(text, source=source)
    return {"status": "ok", "chunks_added": count}


@app.get("/api/kb/stats")
async def kb_stats():
    """知识库统计"""
    return kb.stats()


@app.get("/api/kb/search")
async def kb_search(q: str, top_k: int = 5):
    """搜索知识库"""
    results = kb.search(q, top_k)
    return {"query": q, "results": results}


@app.delete("/api/kb/source/{source}")
async def kb_delete_source(source: str):
    """删除知识库中的某个来源"""
    count = kb.delete_source(source)
    return {"status": "ok", "deleted_chunks": count}


@app.delete("/api/kb")
async def kb_clear():
    """清空知识库"""
    kb.clear()
    return {"status": "ok", "message": "知识库已清空"}


# 挂载前端静态文件
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
