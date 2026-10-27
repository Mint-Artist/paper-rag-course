"""段落级检索评估：用 Open RAG Benchmark 的标注，衡量「检索到的 chunk 有没有覆盖标准答案所在的段落」。

每个问题在 qrels.json 里标注了答案所在的论文（doc_id）和段落（section_id）。
段落原文来自数据集的 corpus/{doc_id}.json（数据集自己从 PDF 抽取的 Markdown 文本），首次运行时自动下载。

怎么判断一个 chunk 命中了标准段落？
  我们的 chunk 来自 PyMuPDF，数据集的段落来自另一个解析器，文字不会逐字相同（公式、空格、断行都不一样），
  所以比较「词三元组（连续 3 个单词）」的重合程度。满足下列任一条件即算命中：
    1. chunk 中至少 30% 的三元组出现在标准段落里（chunk 有相当一部分内容来自该段落）
    2. chunk 包含了标准段落至少 50% 的三元组（段落较短、被整个装进一个大 chunk 的情况）

指标：
  Paper   Recall@K   前 K 个结果里有没有来自正确论文的 chunk（论文级，偏宽松）
  Passage Recall@K   前 K 个结果里有没有命中标准段落的 chunk（段落级，真正决定能不能答对）
  Passage MRR@10     第一个命中段落的 chunk 排名的倒数的平均值（越大说明命中的越靠前）
  上限               索引中「存在」能命中标准段落的 chunk 的问题比例，Passage@K 不会超过它
                     （例如解析时丢弃了参考文献及其后的附录，答案在附录里的问题就永远检索不到）

用法：
    # 1) 评估当前的主索引（storage/chroma，也就是网页正在用的那个）
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --limit 200                 # 只随机抽 200 个问题

    # 2) 对比不同切块方法：在一个论文子集上为每种配置单独建索引（缓存在 storage/eval/，重复运行不会重建）
    python scripts/eval_retrieval.py --compare fixed:800:120 recursive:800:120 recursive:300:50
    python scripts/eval_retrieval.py --compare recursive:800:120 semantic:800:0 --papers 20 --negatives 20

    配置格式为 方法:chunk_size:overlap。子集默认取 30 篇正样本 + 30 篇负样本，每个配置建索引约需几分钟。
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from rag.chunker import CHUNK_METHODS, chunk_paper  # noqa: E402
from rag.embeddings import get_embeddings  # noqa: E402
from rag.indexer import get_vectorstore, load_index_info  # noqa: E402
from rag.parser import load_pdf  # noqa: E402

DATA_DIR = ROOT / "data" / "open_ragbench"
SECTIONS_DIR = DATA_DIR / "sections"
EVAL_CHROMA_DIR = config.STORAGE_DIR / "eval"
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
CORPUS_URL = f"{HF_ENDPOINT}/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv/corpus"
KS = (1, 3, 5, 10)

CHUNK_SHARE = 0.3  # 判定规则 1 的阈值
SECTION_SHARE = 0.5  # 判定规则 2 的阈值


# ---------------------------------------------------------------- 数据准备


def load_queries(papers_dir: Path) -> list[dict]:
    """读取问题，并补上标准段落 id；只保留答案所在论文已下载的问题。"""
    queries = json.loads((DATA_DIR / "queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((DATA_DIR / "qrels.json").read_text(encoding="utf-8"))
    downloaded = {p.stem for p in papers_dir.glob("*.pdf")}
    return [
        {"query_id": qid, **q, "doc_id": qrels[qid]["doc_id"], "section_id": qrels[qid]["section_id"]}
        for qid, q in queries.items()
        if qrels[qid]["doc_id"] in downloaded
    ]


def ensure_sections(doc_ids: set[str]) -> dict[str, dict[int, str]]:
    """下载（并缓存）标准段落文本。原始 corpus 文件里有 base64 图片，较大，这里只保留文字部分。"""
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    missing = sorted(d for d in doc_ids if not (SECTIONS_DIR / f"{d}.json").exists())
    for i, doc_id in enumerate(missing, 1):
        print(f"\r下载标准段落 {i}/{len(missing)}: {doc_id}", end="", flush=True)
        req = urllib.request.Request(f"{CORPUS_URL}/{doc_id}.json", headers={"User-Agent": "paper-rag-course/1.0"})
        for attempt in range(1, 4):  # 网络不稳定时重试
            try:
                corpus = json.loads(urllib.request.urlopen(req, timeout=120).read())
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"\n    第 {attempt} 次失败（{e}），重试 ...")
                time.sleep(5 * attempt)
        sections = {s["section_id"]: s["text"] for s in corpus["sections"]}
        (SECTIONS_DIR / f"{doc_id}.json").write_text(json.dumps(sections, ensure_ascii=False), encoding="utf-8")
    if missing:
        print()
    return {
        d: {int(k): v for k, v in json.loads((SECTIONS_DIR / f"{d}.json").read_text(encoding="utf-8")).items()}
        for d in doc_ids
    }


# ---------------------------------------------------------------- 命中判定


def trigrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z]{2,}|\d+", text.lower())
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def chunk_hits_section(chunk_tri: set, section_tri: set) -> bool:
    if not chunk_tri or not section_tri:
        return False
    common = len(chunk_tri & section_tri)
    return common / len(chunk_tri) >= CHUNK_SHARE or common / len(section_tri) >= SECTION_SHARE


# ---------------------------------------------------------------- 评估


def embed_queries(embeddings, queries: list[dict]) -> list[list[float]]:
    """一次性批量计算所有问题的向量（加上和线上检索相同的 query 指令），多个配置共用。"""
    print(f"计算 {len(queries)} 个问题的向量 ...")
    return embeddings.embed_documents([embeddings.query_instruction + q["query"] for q in queries])


def coverage_ceiling(vectorstore, queries: list[dict], section_tri: dict) -> float:
    """上限：索引里是否「存在」能命中标准段落的 chunk（不管排第几）。

    低于 100% 的部分说明标准段落根本没进索引，例如被解析阶段丢掉了（参考文献之后的附录也会被一起丢弃）。
    """
    covered = 0
    by_doc = collections.defaultdict(list)
    for q in queries:
        by_doc[q["doc_id"]].append(q)
    for doc_id, doc_queries in by_doc.items():
        texts = vectorstore._collection.get(where={"source": f"{doc_id}.pdf"}, include=["documents"])["documents"]
        tris = [trigrams(t) for t in texts]
        for q in doc_queries:
            gold = section_tri.get((doc_id, q["section_id"]), set())
            covered += any(chunk_hits_section(t, gold) for t in tris)
    return covered / len(queries)


def evaluate(vectorstore, queries: list[dict], query_vecs, sections: dict) -> dict:
    section_tri = {(d, s): trigrams(t) for d, secs in sections.items() for s, t in secs.items()}
    paper_hits, passage_hits, rr = collections.Counter(), collections.Counter(), 0.0
    by_source = collections.defaultdict(lambda: [0, 0])  # 按答案依赖的内容类型统计 Passage Recall@5
    for q, vec in zip(queries, query_vecs):
        docs = vectorstore.similarity_search_by_vector(vec, k=max(KS))
        gold = section_tri.get((q["doc_id"], q["section_id"]), set())
        paper_rank = passage_rank = None
        for rank, doc in enumerate(docs, 1):
            if doc.metadata["source"].removesuffix(".pdf") != q["doc_id"]:
                continue
            paper_rank = paper_rank or rank
            if chunk_hits_section(trigrams(doc.page_content), gold):
                passage_rank = rank
                break
        for k in KS:
            paper_hits[k] += bool(paper_rank and paper_rank <= k)
            passage_hits[k] += bool(passage_rank and passage_rank <= k)
        rr += 1 / passage_rank if passage_rank else 0
        by_source[q["source"]][0] += bool(passage_rank and passage_rank <= 5)
        by_source[q["source"]][1] += 1
    n = len(queries)
    return {
        "num_queries": n,
        "ceiling": coverage_ceiling(vectorstore, queries, section_tri),
        "paper_recall": {k: paper_hits[k] / n for k in KS},
        "passage_recall": {k: passage_hits[k] / n for k in KS},
        "passage_mrr@10": rr / n,
        "passage_recall@5_by_source": {s: h / t for s, (h, t) in sorted(by_source.items())},
    }


def build_eval_index(embeddings, spec: str, papers: list[Path]):
    """为一个切块配置建（或复用）评估专用的索引。"""
    method, size, overlap = spec.split(":")
    if method not in CHUNK_METHODS:
        raise SystemExit(f"未知的切块方法 {method}，可选 {CHUNK_METHODS}")
    # 集合名里带上论文列表的哈希：论文子集变了就会重建
    digest = hashlib.md5(",".join(p.stem for p in papers).encode()).hexdigest()[:8]
    name = f"eval-{method}-{size}-{overlap}-{digest}"
    store = get_vectorstore(embeddings, collection_name=name, persist_dir=EVAL_CHROMA_DIR)
    if store._collection.count() > 0:
        print(f"[{spec}] 复用已有的评估索引（{store._collection.count()} 个 chunk）")
        return store

    t0 = time.time()
    chunks = []
    for path in papers:
        pages = load_pdf(path, config.SKIP_REFERENCES)
        if pages:
            chunks.extend(chunk_paper(pages, method, int(size), int(overlap), embeddings))
    print(f"[{spec}] {len(papers)} 篇论文 -> {len(chunks)} 个 chunk，正在向量化 ...")
    for i in range(0, len(chunks), 64):
        store.add_documents(chunks[i : i + 64], ids=[c.id for c in chunks[i : i + 64]])
        print(f"\r      {min(i + 64, len(chunks))}/{len(chunks)}", end="", flush=True)
    print(f"\n[{spec}] 建索引用时 {time.time() - t0:.0f}s")
    return store


def print_table(rows: list[tuple[str, dict]]):
    header = ["配置", "问题数"] + [f"Paper@{k}" for k in (1, 5)] + [f"Passage@{k}" for k in KS] + ["MRR@10", "上限"]
    print("\n" + " | ".join(header))
    print(" | ".join("---" for _ in header))
    for name, r in rows:
        cells = [name, str(r["num_queries"])]
        cells += [f"{r['paper_recall'][k]:.1%}" for k in (1, 5)]
        cells += [f"{r['passage_recall'][k]:.1%}" for k in KS]
        cells += [f"{r['passage_mrr@10']:.3f}", f"{r['ceiling']:.1%}"]
        print(" | ".join(cells))
    print("\n上限 = 索引中存在能命中标准段落的 chunk 的问题比例；Passage@K 不可能超过它。")
    print("\nPassage@5，按答案依赖的内容类型（只抽取文字时，依赖图片/表格的问题通常更难）：")
    for name, r in rows:
        parts = [f"{s}={v:.0%}" for s, v in r["passage_recall@5_by_source"].items()]
        print(f"  {name}: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs="+", metavar="方法:size:overlap", help="对比多个切块配置")
    parser.add_argument("--papers", type=int, default=30, help="--compare 时使用的正样本论文数")
    parser.add_argument("--negatives", type=int, default=30, help="--compare 时使用的负样本论文数（干扰文档）")
    parser.add_argument("--limit", type=int, default=0, help="随机抽取的问题数，0 表示全部")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (DATA_DIR / "qrels.json").exists():
        raise SystemExit("请先运行 python scripts/download_open_ragbench.py 下载数据集")
    queries = load_queries(config.PAPERS_DIR)
    rng = random.Random(args.seed)
    embeddings = get_embeddings()
    rows = []

    if not args.compare:
        # ---------- 模式 1：评估主索引 ----------
        info = load_index_info()
        if not info:
            raise SystemExit("主索引不存在，请先运行 python ingest.py")
        if args.limit:
            queries = rng.sample(queries, min(args.limit, len(queries)))
        sections = ensure_sections({q["doc_id"] for q in queries})
        vecs = embed_queries(embeddings, queries)
        name = f"主索引 {info['chunk_method']}:{info['chunk_size']}:{info['chunk_overlap']}"
        rows.append((name, evaluate(get_vectorstore(embeddings), queries, vecs, sections)))
    else:
        # ---------- 模式 2：在论文子集上对比切块配置 ----------
        positives = sorted({q["doc_id"] for q in queries})
        negatives = sorted({p.stem for p in config.PAPERS_DIR.glob("*.pdf")} - set(positives))
        pos = rng.sample(positives, min(args.papers, len(positives)))
        neg = rng.sample(negatives, min(args.negatives, len(negatives)))
        papers = [config.PAPERS_DIR / f"{d}.pdf" for d in sorted(pos + neg)]
        queries = [q for q in queries if q["doc_id"] in set(pos)]
        if args.limit:
            queries = rng.sample(queries, min(args.limit, len(queries)))
        print(f"子集：{len(pos)} 篇正样本 + {len(neg)} 篇负样本，{len(queries)} 个问题")
        sections = ensure_sections(set(pos))
        vecs = embed_queries(embeddings, queries)
        for spec in args.compare:
            store = build_eval_index(embeddings, spec, papers)
            rows.append((spec, evaluate(store, queries, vecs, sections)))

    print_table(rows)
    out = DATA_DIR / "eval_results" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到 {out}")


if __name__ == "__main__":
    main()
