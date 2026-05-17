"""
学习 Agent — FastAPI 接口
启动: python -m uvicorn api.main:app --reload --port 8000
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

app = FastAPI(title="学习 Agent", version="1.1.0")

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
    return {"status": "ok", "agent": "学习 Agent v1.1.0"}


# ========== 知识库接口 ==========

@app.post("/api/kb/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文档到知识库"""
    allowed = {".pdf", ".md", ".txt", ".py", ".js", ".html", ".json"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"不支持的格式: {ext}，支持: {', '.join(allowed)}")

    # 保存临时文件
    tmp_path = os.path.join(KB_DIR, f"_tmp_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 摄入知识库（用原始文件名作为 source）
    count = kb.ingest_file(tmp_path)
    # 修正 source 名称
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
