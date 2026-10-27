"""命令行构建索引（步骤 1~3）。

用法：
    python ingest.py                                   # 使用 config.py / .env 中的默认参数
    python ingest.py --method fixed --size 500 --overlap 50
    python ingest.py --method semantic --size 1200
    python ingest.py --inspect                         # 只解析+切块，打印前几个 chunk，不写入向量库
"""
import argparse

import config
from rag.chunker import CHUNK_METHODS, chunk_paper
from rag.embeddings import get_embeddings
from rag.parser import load_papers


def inspect(method: str, size: int, overlap: int, n: int = 5):
    """课堂演示用：直观看看不同切块方法切出来的东西长什么样。"""
    papers = load_papers(config.PAPERS_DIR, config.SKIP_REFERENCES)
    embeddings = get_embeddings() if method == "semantic" else None
    for source, pages in papers.items():
        chunks = chunk_paper(pages, method, size, overlap, embeddings)
        print(f"\n===== {source} | {len(pages)} 页 | {len(chunks)} 个 chunk =====")
        for c in chunks[:n]:
            m = c.metadata
            print(f"\n--- #{m['chunk_index']}  p.{m['page_start']}-{m['page_end']}  {len(c.page_content)} 字符 ---")
            print(c.page_content)


def main():
    parser = argparse.ArgumentParser(description="解析 papers/ 下的论文并构建向量索引")
    parser.add_argument("--method", choices=CHUNK_METHODS, default=config.CHUNK_METHOD)
    parser.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--keep-references", action="store_true", help="保留参考文献部分")
    parser.add_argument("--inspect", action="store_true", help="只预览切块结果，不建索引")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.method, args.size, args.overlap)
        return

    from rag.indexer import build_index, get_vectorstore

    embeddings = get_embeddings()
    build_index(
        get_vectorstore(embeddings),
        embeddings,
        method=args.method,
        chunk_size=args.size,
        chunk_overlap=args.overlap,
        skip_references=not args.keep_references,
    )


if __name__ == "__main__":
    main()
