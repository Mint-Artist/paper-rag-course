"""第 2 步：文档切块 —— 使用 LangChain 的 TextSplitter。

做法：把一篇论文的所有页拼成一个长字符串，用 splitter 切分，
再根据每个 chunk 在全文中的起始位置（start_index）反查它来自第几页。
这样 chunk 可以跨页（论文段落经常跨页），页码也不会丢。

提供 3 种方法，课堂上可以对比它们的检索效果：
  fixed      CharacterTextSplitter(separator="")：固定长度滑动窗口，最简单，但会把句子/单词拦腰截断
  recursive  RecursiveCharacterTextSplitter：依次尝试按 段落 > 换行 > 句子 > 空格 切，尽量保持语义完整
  semantic   SemanticChunker：相邻句子的 embedding 相似度骤降处视为话题切换点（会调用 embedding API）
"""
import bisect

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

CHUNK_METHODS = ["fixed", "recursive", "semantic"]
MIN_CHUNK_CHARS = 50

PAGE_SEP = "\n\n"
# 递归切分时依次尝试的分隔符（中英文都考虑）
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "。", "! ", "? ", "！", "？", "; ", "；", ", ", "，", " ", ""]
# 语义切分时的分句规则：英文句末标点后跟空白，或中文句末标点
SENTENCE_SPLIT_REGEX = r"(?<=[.?!])\s+|(?<=[。！？])"


def recursive_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=RECURSIVE_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator="end",  # 句号等分隔符留在前一块的末尾
        add_start_index=True,  # 在 metadata 里记录 chunk 在原文中的起始位置
    )


def split_text(text: str, method: str, chunk_size: int, chunk_overlap: int, embeddings: Embeddings | None):
    """返回 [(chunk 文本, 在 text 中的起始位置)]。"""
    if method == "fixed":
        splitter = CharacterTextSplitter(
            separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        return [(d.page_content, d.metadata["start_index"]) for d in splitter.create_documents([text])]

    if method == "recursive":
        splitter = recursive_splitter(chunk_size, chunk_overlap)
        return [(d.page_content, d.metadata["start_index"]) for d in splitter.create_documents([text])]

    if method == "semantic":
        if embeddings is None:
            raise ValueError("semantic 切分需要传入 embeddings")
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",  # 相邻句子距离超过第 90 百分位处断开
            breakpoint_threshold_amount=90,
            sentence_split_regex=SENTENCE_SPLIT_REGEX,
        )
        # SemanticChunker 不限制块的大小，过长的块再用递归切分兜底
        fallback = recursive_splitter(chunk_size, chunk_overlap)
        pieces = []
        for chunk in splitter.split_text(text):
            pieces.extend([chunk] if len(chunk) <= chunk_size * 1.5 else fallback.split_text(chunk))
        # SemanticChunker 会用空格重新拼接句子，没有准确的 start_index，所以用开头片段在原文里定位
        results, cursor = [], 0
        for piece in pieces:
            found = text.find(piece[:30], cursor)
            cursor = cursor if found == -1 else found
            results.append((piece, cursor))
        return results

    raise ValueError(f"未知的切分方法: {method}，可选 {CHUNK_METHODS}")


def chunk_paper(
    pages: list[Document], method: str, chunk_size: int, chunk_overlap: int, embeddings: Embeddings | None = None
) -> list[Document]:
    """把一篇论文（每页一个 Document）切成若干 chunk Document。"""
    full_text, page_starts, page_nos = "", [], []
    for page in pages:
        if full_text:
            full_text += PAGE_SEP
        page_starts.append(len(full_text))
        page_nos.append(page.metadata["page"])
        full_text += page.page_content

    def page_of(offset: int) -> int:
        return page_nos[max(0, bisect.bisect_right(page_starts, offset) - 1)]

    source, title = pages[0].metadata["source"], pages[0].metadata["title"]
    chunks = []
    pending = None  # 过短的碎片（例如单独一行的章节标题 "Abstract"）先暂存，拼到下一块的开头
    for text, start in split_text(full_text, method, chunk_size, chunk_overlap, embeddings):
        text = text.strip()
        if pending:
            text, start = pending[0] + "\n\n" + text, pending[1]
            pending = None
        if len(text) < MIN_CHUNK_CHARS:
            pending = (text, start)
            continue
        idx = len(chunks)
        chunks.append(
            Document(
                id=f"{source}::{idx}",
                page_content=text,
                metadata={
                    "source": source,
                    "title": title,
                    "page_start": page_of(start),
                    "page_end": page_of(start + len(text) - 1),
                    "chunk_index": idx,
                    "method": method,
                },
            )
        )
    if pending and chunks:  # 末尾剩下的碎片并入最后一块
        chunks[-1].page_content += "\n\n" + pending[0]
        chunks[-1].metadata["page_end"] = page_of(pending[1] + len(pending[0]) - 1)
    return chunks
