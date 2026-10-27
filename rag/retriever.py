"""第 4 步：知识检索 —— 把用户 query 向量化，在向量库里找最相近的 top_k 个 chunk。"""
from dataclasses import dataclass

from langchain_chroma import Chroma

import config

from .indexer import load_index_info


@dataclass
class RetrievedChunk:
    rank: int
    score: float  # 余弦相似度，越大越相关
    chunk_id: str
    text: str
    source: str
    title: str
    page_start: int
    page_end: int


class Retriever:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.vectorstore._collection.count() == 0:
            return []
        info = load_index_info()
        if info and info["embedding_model"] != config.EMBEDDING_MODEL:
            raise RuntimeError(
                f"索引由 {info['embedding_model']} 构建，当前模型是 {config.EMBEDDING_MODEL}，请重建索引"
            )
        # 常见写法是 vectorstore.as_retriever(search_kwargs={"k": top_k})，但它不返回分数；
        # 这里要在前端展示相关性分数，所以直接调用带分数的检索方法
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        return [
            RetrievedChunk(
                rank=i + 1,
                score=round(score, 4),
                chunk_id=doc.id,
                text=doc.page_content,
                source=doc.metadata["source"],
                title=doc.metadata["title"],
                page_start=doc.metadata["page_start"],
                page_end=doc.metadata["page_end"],
            )
            for i, (doc, score) in enumerate(results)
        ]
