"""Embedding 模型：调用硅基流动（SiliconFlow）的 Qwen3-Embedding API。

硅基流动的接口和 OpenAI 兼容，所以直接用 LangChain 的 OpenAIEmbeddings，改一下 base_url 即可。
"""
from langchain_openai import OpenAIEmbeddings

import config


class InstructedEmbeddings(OpenAIEmbeddings):
    """在 query 前加任务指令（Qwen3-Embedding 的推荐用法），文档向量化时不加。"""

    query_instruction: str = ""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(self.query_instruction + text)

    async def aembed_query(self, text: str) -> list[float]:
        return await super().aembed_query(self.query_instruction + text)


def get_embeddings() -> InstructedEmbeddings:
    if not config.SILICONFLOW_API_KEY:
        raise RuntimeError("未设置 SILICONFLOW_API_KEY，请在 .env 中配置")
    return InstructedEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.SILICONFLOW_BASE_URL,
        api_key=config.SILICONFLOW_API_KEY,
        dimensions=config.EMBEDDING_DIMENSIONS,
        chunk_size=config.EMBEDDING_BATCH_SIZE,  # 注意：这里的 chunk_size 指每次请求的文本条数，不是切块大小
        # 必须关掉：否则会先用 OpenAI 的 tiktoken 把文本转成 token id 再发送，非 OpenAI 服务不认
        check_embedding_ctx_length=False,
        query_instruction=config.QUERY_INSTRUCTION,
    )
