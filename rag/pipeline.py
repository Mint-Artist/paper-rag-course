"""把 5 个步骤串起来的 RAG 流水线。前端和命令行脚本都只和这个类打交道。"""
from collections.abc import Iterator
from dataclasses import asdict

import config

from .embeddings import get_embeddings
from .generator import Generator
from .indexer import build_index, get_vectorstore, load_index_info
from .retriever import Retriever


class RAGPipeline:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = get_vectorstore(self.embeddings)
        self.retriever = Retriever(self.vectorstore)
        self.generator = Generator()

    # ---------- 离线：步骤 1~3 ----------
    def build_index(self, **kwargs) -> dict:
        return build_index(self.vectorstore, self.embeddings, **kwargs)

    def index_info(self) -> dict | None:
        info = load_index_info()
        if info is not None:
            info["num_chunks_in_db"] = self.vectorstore._collection.count()
        return info

    # ---------- 在线：步骤 4~5 ----------
    def search(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        return [asdict(c) for c in self.retriever.retrieve(query, top_k)]

    def chat(
        self,
        history: list[dict],
        question: str,
        top_k: int = config.TOP_K,
        rewrite: bool = True,
        thinking: bool = False,
    ) -> Iterator[tuple[str, object]]:
        """一次完整的问答，按顺序产出事件，方便前端实时展示每一步：
        ("query", 检索用的查询) -> ("chunks", 检索结果) -> ("reasoning"/"content", 文本片段)...
        """
        query = self.generator.rewrite_query(history, question) if rewrite else question
        yield "query", query

        chunks = self.retriever.retrieve(query, top_k)
        yield "chunks", [asdict(c) for c in chunks]

        yield from self.generator.stream_answer(history, question, chunks, thinking)
