"""第 1 步：文档解析 —— 用 LangChain 的 DocumentLoader 把论文读成 Document 列表（每页一个）。

LangChain 里所有数据都用 `Document(page_content=..., metadata={...})` 表示，
后面的切块、向量库、检索都围绕它展开。

PyMuPDFLoader 抽出的论文文本还需要清洗：
  * 一行结尾的连字符断词：  "retrie-\\nval"  -> "retrieval"
  * 段落内部的硬换行：每行都有 \\n，需要合并
  * 参考文献：大量作者名/年份，会污染检索结果，可以选择丢弃
"""
import re
from pathlib import Path

import pymupdf
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}

# 单独成行的「参考文献」标题，遇到它之后的内容全部丢弃
_REFERENCES_RE = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography|参考文献)\s*$", re.I | re.M)


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # 行尾连字符断词
    # 句末标点后的换行视为段落边界（保留，供递归切分使用），其余换行是排版造成的，替换为空格
    text = re.sub(r"(?<=[.!?。！？:：])[ \t]*\n\s*", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def guess_title(path: Path) -> str:
    """论文标题通常是第一页字号最大的那几行文字。

    要排除 arXiv 在页面左侧竖排打印的编号水印（"arXiv:2401.03328v2 [econ.TH] 10 Dec 2024"），它的字号往往最大。
    """
    with pymupdf.open(path) as pdf:
        if not len(pdf):
            return ""
        # 以「行」为单位统计字号（取行内最大字号），这样小型大写字母（首字母更大）的标题不会被拆散
        lines = []
        for block in pdf[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                if abs(line["dir"][1]) < 0.01 and len(text) >= 2 and not text.lower().startswith("arxiv"):
                    lines.append((round(max(span["size"] for span in line["spans"]), 1), text))
    # 从大到小尝试各个字号，取第一个「像标题」的：字母够多（排除公式符号），又不太长（排除正文）
    candidates = []
    for size in sorted({s for s, _ in lines}, reverse=True):
        text = re.sub(r"\s+", " ", " ".join(t for s, t in lines if s == size))
        if len(re.findall(r"[A-Za-z一-鿿]", text)) >= 15:
            if len(text) <= 300:
                return text
            candidates.append(text)
    return candidates[0][:150] if candidates else ""


def load_pdf(path: Path, skip_references: bool = True) -> list[Document]:
    # mode="page"：每页一个 Document，metadata 里自带页码 page（从 0 开始）；sort=True 按阅读顺序排列文本
    pages = PyMuPDFLoader(str(path), mode="page", sort=True).load()
    title = (pages[0].metadata.get("title") or "").strip() if pages else ""

    docs = []
    for page in pages:
        text = page.page_content
        ref = _REFERENCES_RE.search(text) if skip_references else None
        if ref:
            text = text[: ref.start()]
        text = clean_text(text)
        if text:
            docs.append(Document(page_content=text, metadata={"page": page.metadata["page"] + 1}))
        if ref:
            break

    # PDF 元数据里的标题经常为空或是无意义的内容，退而求其次从第一页版面里猜
    if len(title) < 8 or title.lower().startswith(("microsoft word", "untitled")):
        title = guess_title(path) or path.stem
    for d in docs:
        d.metadata.update(source=path.name, title=title)
    return docs


def load_text(path: Path) -> list[Document]:
    docs = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
    return [Document(page_content=d.page_content, metadata={"page": 1, "source": path.name, "title": path.stem}) for d in docs]


def load_papers(folder: Path, skip_references: bool = True) -> dict[str, list[Document]]:
    """解析文件夹下所有论文，返回 {文件名: [每页的 Document]}。"""
    papers = {}
    for path in sorted(folder.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            pages = load_pdf(path, skip_references) if path.suffix.lower() == ".pdf" else load_text(path)
        except Exception as e:  # 个别 PDF 损坏或格式特殊时跳过，不影响其他论文
            print(f"      ! 解析失败，已跳过 {path.name}: {e}")
            continue
        if pages:
            papers[path.name] = pages
    return papers
