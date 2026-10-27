"""全局配置：所有可调参数都集中在这里，优先从 .env / 环境变量读取。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---------- 大模型（DeepSeek） ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")  # DeepSeek-V4.1-Flash

# ---------- Embedding 模型（硅基流动 SiliconFlow，OpenAI 兼容接口） ----------
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
# Qwen3-Embedding 支持自定义输出维度（最大 2560），留空表示使用模型默认维度
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS") or 0) or None
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))  # 每次 API 请求包含的文本条数
# Qwen3-Embedding 官方建议：query 加上任务指令，文档不加，检索效果更好
QUERY_INSTRUCTION = os.getenv(
    "QUERY_INSTRUCTION",
    "Instruct: Given a question, retrieve relevant passages from academic papers that answer the question\nQuery: ",
)

# ---------- 路径 ----------
PAPERS_DIR = Path(os.getenv("PAPERS_DIR", ROOT / "papers"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", ROOT / "storage"))
CHROMA_DIR = STORAGE_DIR / "chroma"
INDEX_INFO_FILE = STORAGE_DIR / "index_info.json"
COLLECTION_NAME = "papers"

# ---------- 切块默认参数 ----------
CHUNK_METHOD = os.getenv("CHUNK_METHOD", "recursive")  # fixed / recursive / semantic
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))  # 字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
SKIP_REFERENCES = os.getenv("SKIP_REFERENCES", "true").lower() == "true"

# ---------- 检索与生成 ----------
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))  # 送进 LLM 的历史轮数

# ---------- Web 服务 ----------
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
# 只读模式：用于公网部署，直接使用本机建好的索引，禁止通过网页重建索引
READ_ONLY = os.getenv("READ_ONLY", "false").lower() == "true"
