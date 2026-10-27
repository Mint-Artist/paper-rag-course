"""第 5 步：答案生成 —— 用 LangChain 的 PromptTemplate + ChatDeepSeek 组成链（LCEL），流式生成回答。"""
from collections.abc import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek

import config

from .retriever import RetrievedChunk

SYSTEM_PROMPT = """你是一个学术论文问答助手，负责根据用户本地论文库中检索到的片段回答问题。

规则：
1. 优先依据【参考资料】作答，并在相应句子后用 [编号] 标注出处，例如 "……提升了召回率 [2]"。
2. 如果参考资料不足以回答，请明确说明"在已有论文中没有找到相关信息"，可以补充你的通用知识，但必须注明那部分不是来自论文。
3. 用户用什么语言提问就用什么语言回答；论文术语可保留英文原文。
4. 回答条理清晰，必要时使用 Markdown 列表或表格。"""

# 问答 prompt：系统指令 + 历史对话 + 本轮（参考资料 + 问题）
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你的任务是把多轮对话中用户的最新问题改写成一个「独立、完整、适合做向量检索」的查询语句。
- 补全代词和省略的指代（例如 "它的实验结果呢" -> "XXX 方法的实验结果"）
- 保留论文中的专有名词和英文术语
- 如果最新问题本身已经完整，原样输出
只输出改写后的查询，不要任何解释。""",
        ),
        ("human", "对话历史：\n{dialog}\n\n最新问题：{question}"),
    ]
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"p.{c.page_start}-{c.page_end}"
        parts.append(f"[{c.rank}] 来源：{c.source}（{pages}）\n{c.text}")
    return "\n\n".join(parts) if parts else "（知识库中没有检索到相关内容）"


def thinking(enabled: bool) -> dict:
    """DeepSeek 的思考模式开关，通过 extra_body 透传给 API。"""
    return {"thinking": {"type": "enabled" if enabled else "disabled"}}


class Generator:
    def __init__(self):
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY，请在 .env 中配置")
        self.llm = ChatDeepSeek(
            model=config.LLM_MODEL, api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL
        )
        # LCEL：prompt | 模型 | 输出解析器。查询改写是简单任务，关闭思考、温度设为 0
        self.rewrite_chain = (
            REWRITE_PROMPT | self.llm.bind(temperature=0, extra_body=thinking(False)) | StrOutputParser()
        )

    def rewrite_query(self, history: list[dict], question: str) -> str:
        """多轮对话中的查询改写：没有历史时直接返回原问题。"""
        if not history:
            return question
        dialog = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
        return self.rewrite_chain.invoke({"dialog": dialog, "question": question}).strip() or question

    def stream_answer(
        self, history: list[dict], question: str, chunks: list[RetrievedChunk], enable_thinking: bool = False
    ) -> Iterator[tuple[str, str]]:
        """流式生成，逐段产出 ("reasoning", 文本) 或 ("content", 文本)。

        历史轮次只保留纯文本问答（不带当时的参考资料），既保留多轮语境，又控制 token 数量；
        只有本轮问题会附上新检索到的参考资料。
        """
        chain = ANSWER_PROMPT | self.llm.bind(extra_body=thinking(enable_thinking))
        inputs = {
            "history": history[-config.MAX_HISTORY_TURNS * 2 :],  # [{"role": ..., "content": ...}] 可直接作为消息
            "context": format_context(chunks),
            "question": question,
        }
        for chunk in chain.stream(inputs):
            # ChatDeepSeek 把思考过程放在 additional_kwargs["reasoning_content"] 里
            reasoning = chunk.additional_kwargs.get("reasoning_content")
            if reasoning:
                yield "reasoning", reasoning
            if chunk.content:
                yield "content", chunk.content
