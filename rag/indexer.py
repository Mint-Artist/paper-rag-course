"""第 3 步：chunk 索引 —— 用 embedding 模型把每个 chunk 向量化，写入向量数据库 Chroma。"""
import json
import time
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import config

from .chunker import chunk_paper
from .parser import load_papers


def get_vectorstore(
    embeddings: Embeddings, collection_name: str = config.COLLECTION_NAME, persist_dir: Path = config.CHROMA_DIR
) -> Chroma:
    """LangChain 的 Chroma 向量库：本地持久化，embedding_function 负责把文本转成向量。"""
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        # 用余弦距离（默认是 L2）；这样检索得到的相关性分数 = 1 - 余弦距离 = 余弦相似度
        collection_configuration={"hnsw": {"space": "cosine"}},
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )


def build_index(
    vectorstore: Chroma,
    embeddings: Embeddings,
    method: str = config.CHUNK_METHOD,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    skip_references: bool = config.SKIP_REFERENCES,
    log=print,
) -> dict:
    """完整的离线流程：解析 -> 切块 -> 向量化 -> 入库。返回本次索引的统计信息。"""
    t0 = time.time()

    log(f"[1/3] 解析文档: {config.PAPERS_DIR}")
    papers = load_papers(config.PAPERS_DIR, skip_references)
    if not papers:
        raise RuntimeError(f"{config.PAPERS_DIR} 下没有找到 PDF / MD / TXT 文件")
    for source, pages in papers.items():
        chars = sum(len(p.page_content) for p in pages)
        log(f"      - {source}: {len(pages)} 页, {chars} 字符 | {pages[0].metadata['title'][:60]}")

    log(f"[2/3] 切块: method={method}, size={chunk_size}, overlap={chunk_overlap}")
    chunks: list[Document] = []
    for pages in papers.values():
        chunks.extend(chunk_paper(pages, method, chunk_size, chunk_overlap, embeddings))
    avg_len = sum(len(c.page_content) for c in chunks) / max(1, len(chunks))
    log(f"      共 {len(chunks)} 个 chunk，平均 {avg_len:.0f} 字符")

    log(f"[3/3] 调用 {config.EMBEDDING_MODEL} 向量化并写入 Chroma")
    vectorstore.reset_collection()  # 清空旧索引
    batch = 64
    for i in range(0, len(chunks), batch):
        # add_documents 内部会调用 embeddings.embed_documents 计算向量，再写入 Chroma
        vectorstore.add_documents(chunks[i : i + batch], ids=[c.id for c in chunks[i : i + batch]])
        log(f"      {min(i + batch, len(chunks))}/{len(chunks)}")

    info = {
        "chunk_method": method,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "skip_references": skip_references,
        "embedding_model": config.EMBEDDING_MODEL,
        "num_docs": len(papers),
        "num_chunks": len(chunks),
        "avg_chunk_chars": round(avg_len),
        "documents": [
            {"source": s, "title": p[0].metadata["title"], "pages": len(p)} for s, p in papers.items()
        ],
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }
    config.INDEX_INFO_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"完成，用时 {info['seconds']}s")
    return info


def load_index_info() -> dict | None:
    if config.INDEX_INFO_FILE.exists():
        return json.loads(config.INDEX_INFO_FILE.read_text(encoding="utf-8"))
    return None
