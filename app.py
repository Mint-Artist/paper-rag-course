"""Web 服务：FastAPI 后端 + static/ 下的前端页面。

启动：python app.py   然后浏览器打开 http://127.0.0.1:8000（端口可在 .env 中用 PORT 修改）
"""
import json
import threading

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from rag.chunker import CHUNK_METHODS
from rag.parser import SUPPORTED_SUFFIXES
from rag.pipeline import RAGPipeline

app = FastAPI(title="Paper RAG")
app.mount("/static", StaticFiles(directory=config.ROOT / "static"), name="static")

pipeline = RAGPipeline()
index_lock = threading.Lock()  # 防止重复点击「重建索引」


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]  # 完整对话，最后一条是本轮用户问题
    top_k: int = Field(config.TOP_K, ge=1, le=20)
    rewrite: bool = True
    thinking: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(config.TOP_K, ge=1, le=20)


class IndexRequest(BaseModel):
    method: str = config.CHUNK_METHOD
    chunk_size: int = Field(config.CHUNK_SIZE, ge=100, le=8000)
    chunk_overlap: int = Field(config.CHUNK_OVERLAP, ge=0, le=2000)


@app.get("/")
def index_page():
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p.name for p in config.PAPERS_DIR.glob("**/*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return {
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_methods": CHUNK_METHODS,
        "defaults": {
            "method": config.CHUNK_METHOD,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K,
        },
        "papers_dir": str(config.PAPERS_DIR),
        "files": files,
        "index": pipeline.index_info(),
        "read_only": config.READ_ONLY,
    }


@app.post("/api/index")
def rebuild_index(req: IndexRequest):
    # 普通 def 路由由 FastAPI 放到线程池执行，不会阻塞其他请求
    if config.READ_ONLY:
        raise HTTPException(403, "只读模式下不能重建索引")
    if req.method not in CHUNK_METHODS:
        raise HTTPException(400, f"method 必须是 {CHUNK_METHODS} 之一")
    if not index_lock.acquire(blocking=False):
        raise HTTPException(409, "索引正在构建中，请稍候")
    try:
        return pipeline.build_index(method=req.method, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    finally:
        index_lock.release()


@app.post("/api/search")
def search(req: SearchRequest):
    """只做检索、不调用大模型，方便单独调试检索效果。"""
    try:
        return {"query": req.query, "chunks": pipeline.search(req.query, req.top_k)}
    except RuntimeError as e:
        raise HTTPException(400, str(e))


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(400, "最后一条消息必须来自 user")
    history = [m.model_dump() for m in req.messages[:-1]]
    question = req.messages[-1].content

    def event_stream():
        try:
            for kind, data in pipeline.chat(history, question, req.top_k, req.rewrite, req.thinking):
                yield sse(kind, data)
            yield sse("done", {})
        except Exception as e:  # 把错误也推给前端显示
            yield sse("error", {"message": f"{type(e).__name__}: {e}"})

    # Server-Sent Events：先推送检索结果，再逐字推送回答
    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
