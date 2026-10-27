"""下载 Vectara Open RAG Benchmark 的论文 PDF 和问答数据。

数据集：https://huggingface.co/datasets/vectara/open_ragbench （CC BY-NC 4.0，仅限非商业用途）
  - 1000 篇 arXiv 论文：396 篇「正样本」（是某些问题的标准答案来源）+ 604 篇「难负样本」（干扰文档）
  - 3045 个问题，每个问题都标注了答案所在的论文（qrels）和参考答案（answers）

数据集本身不含 PDF，只给了 arXiv 链接，所以 PDF 要从 arXiv 逐篇下载。
arXiv 要求脚本批量访问时使用 export.arxiv.org 并且限速（约每 3 秒 1 个请求），请不要调小 --delay。

用法：
    python scripts/download_open_ragbench.py                          # 默认：100 篇正样本 + 100 篇负样本（实测约 30 分钟）
    python scripts/download_open_ragbench.py --positives 396 --negatives 604   # 全量 1000 篇
    python scripts/download_open_ragbench.py --positives 20 --negatives 0      # 课堂小样本

可以随时中断，重新运行会跳过已下载的文件。
输出：
    papers/{arxiv_id}.pdf                       论文 PDF
    data/open_ragbench/*.json                   原始的问题、答案、相关性标注、PDF 链接
    data/open_ragbench/eval_queries.json        已下载论文对应的问题（可直接用来测试/评估检索）
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")  # 国内可设为 https://hf-mirror.com
DATASET_URL = f"{HF_ENDPOINT}/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv"
META_FILES = ["queries.json", "qrels.json", "answers.json", "pdf_urls.json"]
USER_AGENT = "paper-rag-course/1.0 (educational, non-commercial)"


def fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_metadata(data_dir: Path) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name in META_FILES:
        path = data_dir / name
        if not path.exists():
            print(f"下载 {name} ...")
            path.write_bytes(fetch(f"{DATASET_URL}/{name}"))
        meta[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def select_papers(meta: dict, n_pos: int, n_neg: int, seed: int) -> list[str]:
    positives = sorted({q["doc_id"] for q in meta["qrels"].values()})
    negatives = sorted(set(meta["pdf_urls"]) - set(positives))
    rng = random.Random(seed)  # 固定随机种子，保证每个学生拿到的子集相同
    return rng.sample(positives, min(n_pos, len(positives))) + rng.sample(negatives, min(n_neg, len(negatives)))


def download_pdf(url: str, path: Path, retries: int = 3) -> bool:
    # arXiv 建议程序化访问走 export.arxiv.org
    url = url.replace("://arxiv.org/", "://export.arxiv.org/")
    for attempt in range(1, retries + 1):
        try:
            data = fetch(url)
            if not data.startswith(b"%PDF"):
                raise ValueError("返回内容不是 PDF")
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(path)  # 写完再改名，中断时不会留下损坏的 PDF
            return True
        except Exception as e:
            print(f"    第 {attempt} 次失败: {e}")
            time.sleep(10 * attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positives", type=int, default=100, help="正样本论文数（最多 396）")
    parser.add_argument("--negatives", type=int, default=100, help="负样本论文数（最多 604）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=3.0, help="两次下载之间的间隔秒数")
    parser.add_argument("--papers-dir", type=Path, default=ROOT / "papers")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "open_ragbench")
    args = parser.parse_args()

    meta = download_metadata(args.data_dir)
    selected = select_papers(meta, args.positives, args.negatives, args.seed)
    args.papers_dir.mkdir(parents=True, exist_ok=True)
    print(f"选中 {len(selected)} 篇论文，保存到 {args.papers_dir}")

    failed = []
    for i, doc_id in enumerate(selected, 1):
        path = args.papers_dir / f"{doc_id}.pdf"
        if path.exists():
            continue
        print(f"[{i}/{len(selected)}] {doc_id}")
        if not download_pdf(meta["pdf_urls"][doc_id], path):
            failed.append(doc_id)
        time.sleep(args.delay)

    # 只保留「答案所在论文已经下载」的问题，作为测试/评估集
    downloaded = {p.stem for p in args.papers_dir.glob("*.pdf")}
    eval_queries = [
        {
            "query_id": qid,
            "query": q["query"],
            "type": q["type"],  # abstractive / extractive
            "source": q["source"],  # 答案依赖的内容：text / text-image / text-table / text-table-image
            "doc_id": meta["qrels"][qid]["doc_id"],
            "answer": meta["answers"][qid],
        }
        for qid, q in meta["queries"].items()
        if meta["qrels"][qid]["doc_id"] in downloaded
    ]
    out = args.data_dir / "eval_queries.json"
    out.write_text(json.dumps(eval_queries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成：papers/ 下共有 {len(downloaded)} 篇 PDF，对应 {len(eval_queries)} 个问题 -> {out}")
    if failed:
        print(f"有 {len(failed)} 篇下载失败（重新运行脚本会重试）：{failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
