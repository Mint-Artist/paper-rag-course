# Paper RAG：本地论文知识库问答

一个用于教学的、基于 **LangChain** 的 RAG（Retrieval-Augmented Generation）系统：把论文 PDF 放进文件夹，
就能在网页上和论文「对话」。

- **左侧**：普通的大模型多轮对话（DeepSeek-V4.1-Flash，流式输出，回答中带 `[1]` `[2]` 引用标注）
- **右侧**：本轮从知识库检索出的片段，包括**排名、相关性分数（余弦相似度）、来源论文与页码、片段原文**
  - 点击回答中的引用标签 → 右侧自动定位到对应片段
  - 点击历史中的某条回答 → 右侧切换回那一轮的检索结果

| 环节 | 使用的组件 |
|---|---|
| 大模型 | DeepSeek `deepseek-flash`（`langchain-deepseek` 的 `ChatDeepSeek`） |
| Embedding | 硅基流动 SiliconFlow 的 `Qwen/Qwen3-Embedding-4B` API（`langchain-openai` 的 `OpenAIEmbeddings`） |
| 向量库 | Chroma，本地持久化（`langchain-chroma`） |
| 文档解析 / 切块 | `PyMuPDFLoader`、`CharacterTextSplitter`、`RecursiveCharacterTextSplitter`、`SemanticChunker` |

模型都通过 API 调用，**不需要 GPU，也不用下载模型**。

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+，建议使用虚拟环境）
pip install -r requirements.txt
# 国内网络可加镜像：pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 2. 配置 API Key
cp .env.example .env        # 然后编辑 .env，填入 DEEPSEEK_API_KEY 和 SILICONFLOW_API_KEY

# 3. 准备论文：从 Release 下载课程数据集（见下文「方式一」），或把自己的 PDF 放进 papers/ 文件夹
curl -L -O https://github.com/Mint-Artist/paper-rag-course/releases/download/v1.0-data/papers.zip && unzip papers.zip
python scripts/download_open_ragbench.py   # 补齐评估用的问题和答案，已有的 PDF 会跳过

# 4. 构建索引（论文较多时建议用命令行；也可以在网页「设置」里点击「重建索引」）
python ingest.py

# 5. 启动网页
python app.py               # 浏览器打开 http://127.0.0.1:8000
```

## 课程数据集：Open RAG Benchmark

课程使用 Vectara 发布的 [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench)
（许可 **CC BY-NC 4.0**，仅限非商业用途，使用时请注明来源）：

- 1000 篇 arXiv 论文 PDF，覆盖 arXiv 各学科：396 篇**正样本**（是某些问题的答案来源）+ 604 篇**难负样本**（干扰文档）
- 3045 个问题，每个问题都标注了答案所在的论文和参考答案；约 1/3 的问题需要看论文里的表格或图片才能回答

### 方式一：从本仓库的 Release 下载（推荐，几分钟）

[Release `v1.0-data`](https://github.com/Mint-Artist/paper-rag-course/releases/tag/v1.0-data) 里有两个附件：

| 文件 | 内容 | 解压到 |
|---|---|---|
| `papers.zip`（约 467MB） | 默认的 200 篇论文 PDF | 项目根目录，得到 `papers/*.pdf` |
| `index_qwen3-4b_recursive-800-120.zip`（约 300MB） | 用这 200 篇建好的向量索引（Qwen3-Embedding-4B，`recursive` 800/120） | 项目根目录，得到 `storage/chroma/` 和 `storage/index_info.json` |

```bash
curl -L -O https://github.com/Mint-Artist/paper-rag-course/releases/download/v1.0-data/papers.zip
unzip papers.zip
# 可选：跳过约 18 分钟的建索引，下载后直接 python app.py 就能提问
curl -L -O https://github.com/Mint-Artist/paper-rag-course/releases/download/v1.0-data/index_qwen3-4b_recursive-800-120.zip
unzip index_qwen3-4b_recursive-800-120.zip
# 生成 data/open_ragbench/eval_queries.json 等评估文件（已有的 PDF 会自动跳过，只下载几 MB 的元数据）
python scripts/download_open_ragbench.py
```

注意：预建索引由 `chromadb==1.5.9` 生成，Chroma 大版本不同可能读不了，遇到报错就重新 `python ingest.py`
或者 `pip install chromadb==1.5.9`。想练习切块和建索引的同学，只下载 `papers.zip` 就够了。

### 方式二：从 arXiv 下载

数据集本身不含 PDF，`scripts/download_open_ragbench.py` 会按它提供的链接，从 arXiv 限速（每 3 秒一篇）下载：

```bash
python scripts/download_open_ragbench.py                                   # 默认 200 篇：100 正样本 + 100 负样本
python scripts/download_open_ragbench.py --positives 20 --negatives 20     # 课堂上对比切块方法用的小子集
python scripts/download_open_ragbench.py --positives 396 --negatives 604   # 全量 1000 篇
```

实测参考（默认 200 篇）：下载约 30 分钟，共 533MB；用 `recursive(800/120)` 切块得到约 1.9 万个 chunk，
建索引（调用 embedding API）约 18 分钟。全量 1000 篇的耗时大约是它的 5 倍，所以课上对比切块方法时请用小子集。

随机种子是固定的，所以每个学生下载到的子集都一样；中途中断后重新运行，会跳过已经下载的文件。
下载完成后，`data/open_ragbench/eval_queries.json` 会列出这批论文对应的问题、参考答案和答案所在论文，
可以直接复制到网页里提问，再对照右侧的检索结果，看答案所在论文有没有被检索出来。

一个值得在课上讨论的现象：在这 200 篇论文上随机抽 100 个问题，**答案所在的论文**几乎总能排在第 1 位
（论文之间学科差异很大，找对论文并不难）；但**真正含有答案的那一段**不一定进入 Top-5，
这时模型会说明「论文片段中没有提到」。所以评估切块和检索方法时，要看段落级别的命中，而不只是论文级别。

## 检索评估：`scripts/eval_retrieval.py`

Open RAG Benchmark 为每个问题标注了答案所在的**段落**（`qrels.json` 中的 `section_id`），
评估脚本据此统计：检索到的 chunk 有没有覆盖标准答案所在的段落。

```bash
# 评估当前主索引（网页正在用的那个）
python scripts/eval_retrieval.py

# 对比切块配置：在论文子集上为每个配置单独建索引（缓存在 storage/eval/，重复运行不会重建）
python scripts/eval_retrieval.py --compare fixed:800:120 recursive:800:120 recursive:400:60 --papers 20 --negatives 20
```

- **命中判定**：我们的 chunk（PyMuPDF 抽取）和数据集的段落（另一个解析器抽取）文字不会逐字相同，所以比较「词三元组」的重合度。
  满足任一条件即算命中：chunk 中 ≥30% 的三元组来自标准段落，或者 chunk 包含标准段落 ≥50% 的三元组。
  首次运行时会下载这批论文的标准段落文本（只保留文字，约 8MB）。
- **输出指标**：
  - `Paper@K`：论文级命中率。
  - `Passage@K`：段落级命中率。
  - `MRR@10`：第一个命中段落的 chunk 排名倒数的平均值，越大说明命中越靠前。
  - `上限`：索引中**存在**能命中标准段落的 chunk 的问题比例。
  - 另外还会按问题类型（纯文字 / 依赖图片 / 依赖表格）分组统计。
- 结果同时保存为 JSON，放在 `data/open_ragbench/eval_results/`。

当前主索引（200 篇，recursive 800/120，全部 770 个问题）的实测结果：

| Paper@1 | Passage@1 | Passage@5 | Passage@10 | MRR@10 | 上限 |
|---|---|---|---|---|---|
| 96.8% | 51.4% | 75.6% | 81.6% | 0.622 | 88.3% |

可以看出：**找对论文很容易，找对段落难得多**。另外，上限只有 88% 的主要原因是解析阶段丢弃了 References
**及其之后的全部内容**，附录也一起丢了；而约 12% 的问题答案正好在附录里。这也是一个很好的课堂练习：
改进 `rag/parser.py`，只去掉参考文献、保留附录，再看上限和 Passage@K 的变化。

切块对比的实测示例（20 篇正样本 + 20 篇负样本，154 个问题，四个配置共用时约 12 分钟）：

| 配置 | Passage@1 | Passage@5 | Passage@10 | MRR@10 | 上限 |
|---|---|---|---|---|---|
| fixed:800:120 | 51.3% | 72.7% | 79.9% | 0.607 | 90.9% |
| recursive:800:120 | 53.9% | 77.9% | 85.1% | 0.645 | 93.5% |
| recursive:400:60 | 55.2% | 79.2% | 85.7% | 0.650 | 94.2% |
| recursive:1600:200 | 48.7% | 72.7% | 81.2% | 0.599 | 87.7% |

解读时注意两点：
- 154 个问题的样本较小，相差几个百分点可能只是随机波动；要下结论，就增加 `--papers` 或者换 `--seed` 多跑几次。
- 命中判定是近似的：chunk 越大，重合比例越容易被稀释，所以大 chunk 的分数会略被低估。
  另外，chunk 越小，检索越精准，但每个片段的上下文也越少，送进大模型后不一定答得更好。检索指标只是其中一面。

## 项目结构

```
├── papers/              ← 论文 PDF 放在这里（也支持 .md / .txt）
├── scripts/
│   ├── download_open_ragbench.py  ← 下载课程数据集
│   └── eval_retrieval.py          ← 段落级检索评估 / 切块方法对比
├── data/open_ragbench/  ← 下载脚本生成：问题、参考答案、相关性标注
├── config.py            ← 所有可调参数（模型、路径、切块参数、Top-K……）
├── rag/
│   ├── parser.py        ← 步骤1 文档解析（PyMuPDFLoader + 文本清洗）
│   ├── chunker.py       ← 步骤2 文档切块（3 种 TextSplitter）
│   ├── embeddings.py    ← Embedding 模型（OpenAIEmbeddings 对接硅基流动）
│   ├── indexer.py       ← 步骤3 chunk 索引（向量化 + 写入 Chroma）
│   ├── retriever.py     ← 步骤4 知识检索
│   ├── generator.py     ← 步骤5 答案生成（ChatPromptTemplate | ChatDeepSeek）
│   └── pipeline.py      ← 把 5 步串起来
├── ingest.py            ← 命令行建索引 / 预览切块效果
├── app.py               ← FastAPI 后端
├── static/              ← 前端页面（原生 HTML/CSS/JS，无需构建）
└── storage/             ← 运行后自动生成：向量库与索引信息
```

## RAG 流程与对应代码

```
                ┌──────────────── 离线：建索引（ingest.py） ────────────────┐
  papers/*.pdf → ① 解析 parser.py → ② 切块 chunker.py → ③ 向量化+入库 indexer.py → Chroma
                └──────────────────────────────────────────────────────────┘
                ┌──────────────── 在线：问答（app.py） ─────────────────────┐
  用户问题 →（多轮时先改写查询）→ ④ 检索 retriever.py → Top-K chunks ──→ 右侧面板
                                                    ↓
                                     ⑤ 拼 prompt + LLM 生成 generator.py → 左侧对话
                └──────────────────────────────────────────────────────────┘
```

LangChain 里所有文本都用 `Document(page_content=..., metadata={...})` 表示：
解析得到「每页一个 Document」，切块得到「每个 chunk 一个 Document」，向量库存的、检索返回的也都是 Document。

### ① 文档解析 `rag/parser.py`
`PyMuPDFLoader(mode="page")` 按页读取，然后做论文 PDF 常见的清洗：
合并行尾连字符断词（`retrie-\nval` → `retrieval`）、合并排版造成的段内换行、
**默认丢弃 References 之后的内容**（参考文献里全是人名和年份，会严重干扰检索；可用 `--keep-references` 保留）。

### ② 文档切块 `rag/chunker.py`
先把一篇论文的所有页拼成全文再切，chunk 可以跨页（论文段落经常跨页）；
切块时打开 `add_start_index=True`，根据每个 chunk 在全文中的起始位置反查页码。

| 方法 | LangChain 组件 | 优点 | 缺点 |
|---|---|---|---|
| `fixed` | `CharacterTextSplitter(separator="")` | 最简单、块大小均匀 | 会把句子甚至单词截断 |
| `recursive`（默认） | `RecursiveCharacterTextSplitter` | 依次按 段落 → 换行 → 句子 → 空格 切，尽量保持语义完整 | 块大小不完全均匀 |
| `semantic` | `SemanticChunker`（langchain-experimental） | 在相邻句子 embedding 相似度骤降处断开，按话题切分 | 每个句子都要调用 embedding API，建索引慢、有费用 |

两个细节（也是很好的课堂讨论点）：
- 递归切分遇到「短标题 + 超长段落」时，会把标题单独切成一小块。代码里把这类过短的碎片合并到下一块的开头。
- `SemanticChunker` 不限制块的大小，过长的块会再用递归切分兜底。

用命令行直观对比切块结果（不写入向量库）：
```bash
python ingest.py --inspect --method fixed --size 500 --overlap 50
python ingest.py --inspect --method recursive
python ingest.py --inspect --method semantic --size 1200
```

### ③ chunk 索引 `rag/embeddings.py` + `rag/indexer.py`
- **Embedding**：硅基流动的 `Qwen/Qwen3-Embedding-4B`（2560 维，多语言，支持「中文提问检索英文论文」）。
  硅基流动的接口兼容 OpenAI，所以直接用 `OpenAIEmbeddings`，改 `base_url` 即可。注意两点：
  - 必须设置 `check_embedding_ctx_length=False`，否则 LangChain 会先用 OpenAI 的 tiktoken 把文本转成 token id 再发送，非 OpenAI 服务不认。
  - Qwen3-Embedding 建议**给 query 加任务指令、文档不加**，所以写了一个子类，只重写 `embed_query`。
- **向量库**：`langchain_chroma.Chroma`，本地持久化在 `storage/chroma`，距离度量设为 cosine。

### ④ 知识检索 `rag/retriever.py`
用 `similarity_search_with_relevance_scores` 取 Top-K，**分数 = 1 − 余弦距离 = 余弦相似度**（越接近 1 越相关）。
（LangChain 常见写法 `vectorstore.as_retriever()` 不返回分数，而我们要在前端展示分数。）

**多轮查询改写**：第二轮起，像「它的实验结果怎么样？」这种问题直接检索效果很差，
所以先用一条 LCEL 链（`REWRITE_PROMPT | llm | StrOutputParser()`）结合对话历史把它改写成独立的查询，
右侧面板顶部会显示实际用于检索的查询。可在设置中关闭以对比效果。

### ⑤ 答案生成 `rag/generator.py`
`ChatPromptTemplate` 由三部分组成：系统指令、`MessagesPlaceholder("history")` 历史对话、本轮的参考资料加问题。
链写成 `ANSWER_PROMPT | ChatDeepSeek`，用 `.stream()` 流式输出。
模型要基于资料回答、用 `[编号]` 标注出处，资料不足时要明确说明。
多轮对话只把历史的纯文本问答传给模型（不重复附带历史检索结果），以控制 token 消耗。
勾选「深度思考」会通过 `extra_body` 开启 DeepSeek 的思考模式；
思考过程在流式块的 `additional_kwargs["reasoning_content"]` 里，可在回答上方展开查看。

## 接口（方便学生自己调试）

| 接口 | 说明 |
|---|---|
| `GET  /api/status` | 模型、论文文件、索引信息 |
| `POST /api/index`  | 重建索引 `{"method": "recursive", "chunk_size": 800, "chunk_overlap": 120}` |
| `POST /api/search` | 只检索不生成 `{"query": "...", "top_k": 5}` |
| `POST /api/chat`   | 检索 + 流式生成（SSE），事件依次为 `query` → `chunks` → `reasoning`/`content`… → `done` |

```bash
curl -s localhost:8000/api/search -H 'Content-Type: application/json' \
     -d '{"query":"What datasets are used?","top_k":3}' | python -m json.tool
```

## 部署到服务器（只读模式）

服务器上不重新解析和建索引，而是直接使用本机建好的索引（`storage/chroma` + `storage/index_info.json`）。
在服务器的 `.env` 中设置 `READ_ONLY=true`，网页会隐藏「重建索引」，`/api/index` 返回 403，访客无法触发重新 embedding。
注意：用户每次提问时，问题本身仍需调用一次 embedding API。

```bash
# 1. 上传代码和索引（不传 PDF、评估缓存和本地虚拟环境）
rsync -az -e "ssh -p <SSH端口> -i <私钥>" --exclude .venv/ --exclude '__pycache__/' --exclude 'papers/*.pdf' \
      --exclude data/ --exclude storage/eval/ --exclude .env ./ root@<服务器>:/opt/paper-rag/

# 2. 服务器上：安装与本机完全相同版本的依赖（Chroma 版本不同会读不了索引）
cd /opt/paper-rag && python3 -m venv .venv && .venv/bin/pip install -r deploy/requirements.lock

# 3. 服务器上：配置 .env（API Key、HOST=0.0.0.0、PORT、READ_ONLY=true），然后用 systemd 托管
cp deploy/paper-rag.service /etc/systemd/system/ && systemctl daemon-reload && systemctl enable --now paper-rag
```

更新代码或索引后，重新执行第 1 步，再运行 `systemctl restart paper-rag`；查看日志用 `journalctl -u paper-rag -f`。
本地改过依赖的话，先用 `uv pip freeze > deploy/requirements.lock` 更新版本清单。

## 常见问题

- **端口 8000 被占用**：在 `.env` 里设置 `PORT=8001`。
- **换了 embedding 模型或维度后检索报错**：不同模型的向量维度不同，必须重建索引（页面顶部会提示）。
- **建索引时报 401 / 403**：检查 `.env` 中的 `SILICONFLOW_API_KEY`；报 429 说明请求太频繁，可调小 `EMBEDDING_BATCH_SIZE`。
- **`semantic` 切块很慢**：它要对每个句子调用 embedding API，论文多时建议先用 `recursive`。

## 课堂练习建议

1. **切块对比**：同一组问题，分别用 `fixed(300/0)`、`recursive(800/120)`、`semantic` 建索引，观察右侧分数和片段质量的变化。
2. **chunk_size 的取舍**：太小 → 片段缺上下文；太大 → 分数被稀释、prompt 变长。试试 300 / 800 / 2000。
3. **Query 指令的作用**：把 `.env` 里的 `QUERY_INSTRUCTION` 设为空字符串，比较同一问题的检索分数和排序。
4. **向量维度**：设置 `EMBEDDING_DIMENSIONS=512` 后重建索引，比较检索效果与存储大小。
5. **查询改写**：关闭「多轮查询改写」后追问「它用了哪些数据集？」，看检索结果的差异。
6. **进阶**（自行实现）：BM25 + 向量混合检索（`EnsembleRetriever`）、Reranker 重排序（硅基流动也提供
   `Qwen/Qwen3-Reranker-4B` 等 rerank API）、相似度阈值过滤、为 chunk 加上论文标题/章节作为上下文前缀。
