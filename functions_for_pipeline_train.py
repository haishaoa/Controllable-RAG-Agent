import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

try:
    from helper_functions import escape_quotes
except Exception:

    def escape_quotes(text: str) -> str:
        """备用转义函数,在 helper_functions 不可用时转义双引号。"""

        return text.replace('"', '\\"') if isinstance(text, str) else text


class PlanExecute(TypedDict, total=False):
    curr_state: str
    question: str
    anonymized_question: str
    query_to_retrieve_or_answer: str
    plan: List[str]
    past_steps: List[str]
    mapping: dict
    curr_context: str
    aggregated_context: str
    tool: str
    response: str


class Plan(BaseModel):
    steps: List[str] = Field(description="要遵循的不同步骤,应按顺序排列。")


class AnonymizeQuestion(BaseModel):
    anonymized_question: str = Field(description="匿名化后的问题。")
    mapping: dict = Field(description="变量到原始实体的映射关系。")
    explanation: str = Field(description="简要说明做了哪些匿名化处理。")


class DeAnonymizePlan(BaseModel):
    plan: List[str] = Field(description="变量被替换为原始实体后的计划列表。")


class TaskHandlerOutput(BaseModel):
    query: str = Field(description="用于检索的查询,或用于基于上下文回答的问题。")
    curr_context: str = Field(
        description="如果选择 answer_from_context,则这里是回答所依据的上下文；否则为空字符串。"
    )
    tool: str = Field(
        description="只能是 retrieve_chunks、retrieve_summaries、retrieve_quotes、answer_from_context。"
    )


class KeepRelevantContent(BaseModel):
    relevant_content: str = Field(
        description="从检索到的文档中提取的与查询相关的内容。"
    )


class IsDistilledContentGroundedOnContent(BaseModel):
    grounded: bool = Field(description="提炼内容是否完全基于原始上下文。")
    explanation: str = Field(description="判断理由。")


class IsGroundedOnFacts(BaseModel):
    grounded_on_facts: bool = Field(description="答案是否基于给定上下文。")


class AnswerFromContext(BaseModel):
    answer: str = Field(description="根据上下文生成的答案。")


class CanBeAnsweredAlready(BaseModel):
    can_be_answered: bool = Field(
        description="根据给定上下文是否已经可以完整回答问题。"
    )


class QualitativeRetrievalGraphState(TypedDict, total=False):
    question: str
    context: str
    relevant_context: str


_COMPONENTS: Optional[Dict[str, Any]] = None


def _require_env(name: str, purpose: str) -> str:
    """读取必要环境变量,缺失时给出明确的启动提示。"""

    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"没有读取到 {name}。{purpose}\n"
            f"请先在终端执行：export {name}='你的 key',然后重新启动 Streamlit。"
        )
    return value


def create_llm():
    """创建 DeepSeek Chat 模型。"""

    deepseek_api_key = _require_env(
        "DEEPSEEK_API_KEY",
        "当前项目使用 DeepSeek 作为大模型。",
    )

    return ChatOpenAI(
        temperature=0,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2000")),
        api_key=deepseek_api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )


def create_embeddings():
    """创建 OpenAI Embeddings,用于匹配原 1536 维 FAISS 向量库。"""

    openai_api_key = _require_env(
        "OPENAI_API_KEY",
        "当前项目要继续使用之前的 1536 维 FAISS 向量库,因此查询时也需要 OpenAIEmbeddings。",
    )

    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        openai_api_key=openai_api_key,
    )


def _assert_vector_dim(
    vector_store: FAISS, embeddings: OpenAIEmbeddings, store_name: str
):
    """检查 FAISS 索引维度和当前 embedding 输出维度是否一致。"""

    try:
        index_dim = int(vector_store.index.d)
        query_dim = len(embeddings.embed_query("dimension check"))
    except Exception:
        return

    if index_dim != query_dim:
        raise ValueError(
            f"本地向量库 {store_name} 的维度是 {index_dim},"
            f"但当前 embedding 模型输出维度是 {query_dim}。\n"
            f"这说明这个 FAISS 向量库不是用当前 embedding 模型生成的。\n"
            f"如果要继续使用旧向量库,请设置与建库时一致的 OPENAI_EMBEDDING_MODEL；"
            f"如果要换 embedding 模型,则必须重新构建向量库。"
        )


def create_retrievers():
    """加载本地 FAISS 向量库,并创建 retriever。"""

    embeddings = create_embeddings()

    chunks_vector_store = FAISS.load_local(
        "chunks_vector_store",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    _assert_vector_dim(chunks_vector_store, embeddings, "chunks_vector_store")

    chapter_summaries_vector_store = FAISS.load_local(
        "chapter_summaries_vector_store",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    _assert_vector_dim(
        chapter_summaries_vector_store,
        embeddings,
        "chapter_summaries_vector_store",
    )

    book_quotes_vectorstore = FAISS.load_local(
        "book_quotes_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    _assert_vector_dim(book_quotes_vectorstore, embeddings, "book_quotes_vectorstore")

    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 1})
    chapter_summaries_query_retriever = chapter_summaries_vector_store.as_retriever(
        search_kwargs={"k": 1}
    )
    book_quotes_query_retriever = book_quotes_vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    return (
        chunks_query_retriever,
        chapter_summaries_query_retriever,
        book_quotes_query_retriever,
    )


def _parser_for(model_cls):
    """根据 Pydantic 模型创建 JSON 输出解析器。"""

    return JsonOutputParser(pydantic_object=model_cls)


def _format_instructions(parser: JsonOutputParser) -> str:
    """取得解析器生成的 JSON 格式说明。"""

    return parser.get_format_instructions()


def _chain(prompt_template: str, input_variables: List[str], model_cls):
    """构建 prompt | llm | JsonOutputParser,不使用 with_structured_output。"""

    parser = _parser_for(model_cls)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=input_variables,
        partial_variables={"format_instructions": _format_instructions(parser)},
    )
    return prompt | create_llm() | parser


def create_plan_chain():
    """创建初始规划链,把用户问题拆成可执行步骤。"""

    prompt = """对于给定的查询 {question},请提供一个简单的分步计划,说明如何得出答案。

要求：
1. 计划应包含具体任务,若执行得当,将得出正确答案。
2. 不要添加多余步骤。
3. 最后一步的结果应该是最终答案。
4. 每一步都必须包含必要信息,不要跳过步骤。
5. 必须只输出 json,不要输出解释性文字。

{format_instructions}
"""
    return _chain(prompt, ["question"], Plan)


def create_anonymize_question_chain():
    """创建问题匿名化链,将实体替换为变量并保留映射。"""

    prompt = """你是一个问题匿名化工具。你接收到的问题是：{question}

你的目标：
1. 将输入中的命名实体替换成变量,比如 X、Y、Z。
2. 记录变量到原始实体的映射关系。
3. 如果没有明显命名实体,可以保持问题基本不变,mapping 返回空字典。
4. 必须只输出 json,不要输出解释性文字。

示例：
输入：谁是哈利波特？
输出：{{"anonymized_question":"谁是X？","mapping":{{"X":"哈利波特"}},"explanation":"将哈利波特替换为X"}}

{format_instructions}
"""
    return _chain(prompt, ["question"], AnonymizeQuestion)


def create_deanonymize_plan_chain():
    """创建计划去匿名化链,把变量还原成原始实体。"""

    prompt = """你收到一个任务列表：{plan}
其中一些词可能被替换为了变量。你还收到变量到原始词的映射：{mapping}

请将任务列表中的变量替换为原始词。
如果任务列表中不存在变量,则返回原始任务列表。
必须只输出 json,不要输出解释性文字。

{format_instructions}
"""
    return _chain(prompt, ["plan", "mapping"], DeAnonymizePlan)


def create_break_down_plan_chain():
    """创建计划拆解链,把计划转换为 RAG 工具可执行任务。"""

    prompt = """你收到一个计划：{plan}

请把它优化为更适合 RAG Agent 执行的步骤。
每一步必须能由以下一种方式完成：
1. 从书籍正文 chunks 向量库检索相关信息。
2. 从章节摘要向量库检索相关信息。
3. 从书籍引文向量库检索相关信息。
4. 根据已有上下文回答问题。

要求：
1. 每一步都要明确、可执行。
2. 不要输出空计划。
3. 必须只输出 json,不要输出解释性文字。

{format_instructions}
"""
    return _chain(prompt, ["plan"], Plan)


def create_task_handler_chain():
    """创建任务分发链,为当前任务选择检索或回答工具。"""

    prompt = """你是一名任务处理者。你收到当前任务：{curr_task}

已有聚合上下文：
{aggregated_context}

上一次使用的工具：{last_tool}
过去完成的步骤：{past_steps}
用户原始问题：{question}

你必须从下面四个工具中选择一个：
1. retrieve_chunks：当任务需要从书籍正文 chunks 中检索。
2. retrieve_summaries：当任务需要从章节摘要中检索。
3. retrieve_quotes：当任务需要从书籍引文/原文摘录中检索。
4. answer_from_context：当已有上下文足以回答当前任务。

输出要求：
1. tool 必须严格等于 retrieve_chunks、retrieve_summaries、retrieve_quotes 或 answer_from_context。
2. 如果选择检索工具,query 写适合检索的查询,curr_context 为空字符串。
3. 如果选择 answer_from_context,query 写要回答的问题,curr_context 写回答依据上下文。
4. 必须只输出 json,不要输出解释性文字。

{format_instructions}
"""
    return _chain(
        prompt,
        ["curr_task", "aggregated_context", "last_tool", "past_steps", "question"],
        TaskHandlerOutput,
    )


def create_keep_only_relevant_content_chain():
    """创建相关内容筛选链,从检索文档中提取有用片段。"""

    prompt = """你收到一个查询：{query}
        以及从向量库检索到的文档：
        {retrieved_documents}

        请只保留与查询相关、能够帮助回答问题的内容。
        不要添加检索文档之外的新信息。
        如果文档中没有相关信息,relevant_content 返回空字符串。
        必须只输出 json,不要输出解释性文字。

        {format_instructions}
    """
    return _chain(prompt, ["query", "retrieved_documents"], KeepRelevantContent)


def create_is_distilled_content_grounded_on_content_chain():
    """创建提炼内容校验链,判断提炼结果是否忠于原文。"""

    prompt = """你将收到提炼后的内容：
{distilled_content}

以及原始上下文：
{original_context}

判断提炼后的内容是否完全基于原始上下文。
如果提炼内容没有添加原始上下文以外的信息,grounded 为 true。
否则 grounded 为 false。
必须只输出 json,不要输出解释性文字。

{format_instructions}
"""
    return _chain(
        prompt,
        ["distilled_content", "original_context"],
        IsDistilledContentGroundedOnContent,
    )


def create_answer_question_from_context_chain():
    """创建上下文问答链,只基于给定上下文生成答案。"""

    prompt = """你需要只根据给定上下文回答问题。
        问题：{question}

        上下文：
        {context}

        要求：
        1. 只能使用上下文中的信息。
        2. 如果上下文无法回答,请说明“根据给定上下文无法确定”。
        3. 必须只输出 json,不要输出解释性文字。

        {format_instructions}
    """
    return _chain(prompt, ["question", "context"], AnswerFromContext)


def create_is_grounded_on_facts_chain():
    """创建答案事实性校验链,检查答案是否由上下文支撑。"""

    prompt = """你是一名事实核查员。
请判断答案是否基于给定上下文。

上下文：
{context}

答案：
{answer}

如果答案只使用了上下文信息,grounded_on_facts 为 true。
如果答案包含上下文没有提供的信息,grounded_on_facts 为 false。
必须只输出 json,不要输出解释性文字。

{format_instructions}
"""
    return _chain(prompt, ["context", "answer"], IsGroundedOnFacts)


def create_replanner_chain():
    """创建重规划链,根据已有上下文更新后续计划。"""

    prompt = """针对给定目标,更新后续计划。

目标问题：
{question}

当前剩余计划：
{plan}

已经完成的步骤：
{past_steps}

已经掌握的上下文：
{aggregated_context}

要求：
1. 如果还需要更多检索或回答步骤,请返回后续步骤。
2. 不要重复已经完成的步骤。
3. 如果上下文已经足够回答问题,可以返回空 steps：[]。
4. 必须只输出 json,不要输出解释性文字。

{format_instructions}
"""
    return _chain(
        prompt, ["question", "plan", "past_steps", "aggregated_context"], Plan
    )


def create_can_be_answered_already_chain():
    """创建可回答性判断链,评估当前上下文是否足够回答问题。"""

    prompt = """你收到一个问题：{question}
以及一个上下文：
{context}

你需要判断：仅凭这个上下文,是否已经可以完整回答问题。
如果可以,can_be_answered 为 true。
如果不可以,can_be_answered 为 false。
必须只输出 json,不要输出解释性文字。

{format_instructions}
"""
    return _chain(prompt, ["question", "context"], CanBeAnsweredAlready)


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    """从 dict 或对象中安全读取字段,没有时返回默认值。"""

    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_steps(plan_output: Any) -> List[str]:
    """兼容 dict、Pydantic 对象、list、字符串等结构,统一取出计划步骤。"""

    if plan_output is None:
        return []

    if isinstance(plan_output, list):
        return [str(item).strip() for item in plan_output if str(item).strip()]

    if isinstance(plan_output, str):
        text = plan_output.strip()
        return [text] if text else []

    if isinstance(plan_output, dict):
        raw_steps = plan_output.get("steps", plan_output.get("plan", []))
        return _get_steps(raw_steps)

    if hasattr(plan_output, "steps"):
        return _get_steps(getattr(plan_output, "steps"))

    if hasattr(plan_output, "plan"):
        return _get_steps(getattr(plan_output, "plan"))

    return []


def _normalize_tool_name(tool: str) -> str:
    """把模型可能输出的别名统一成内部工具名。"""

    tool = (tool or "").strip()
    mapping = {
        "retrieve_book_quotes": "retrieve_quotes",
        "answer": "answer_from_context",
        "工具A": "retrieve_chunks",
        "工具B": "retrieve_summaries",
        "工具C": "retrieve_quotes",
        "工具D": "answer_from_context",
    }
    return mapping.get(tool, tool)


def _retrieve_docs(retriever: Any, question: str):
    """兼容新版 invoke 和旧版 get_relevant_documents 的检索调用。"""

    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)
    return retriever.get_relevant_documents(question)


def _run_sub_workflow_and_get_last_state(
    workflow_app, inputs: Dict[str, Any]
) -> Dict[str, Any]:
    """执行子工作流并返回流式输出中的最后一个 state。"""

    last_value = None
    for output in workflow_app.stream(inputs):
        for _, value in output.items():
            last_value = value
    if not isinstance(last_value, dict):
        raise ValueError("子工作流没有返回有效的 state。")
    return last_value


def _append_context(state: PlanExecute, text: Any) -> PlanExecute:
    """把新内容追加到聚合上下文中,忽略空内容。"""

    if not state.get("aggregated_context"):
        state["aggregated_context"] = ""
    text = "" if text is None else str(text).strip()
    if text:
        state["aggregated_context"] += "\n" + text
    return state


def _init_state(state: PlanExecute) -> PlanExecute:
    """为 Agent state 补齐后续节点需要的默认字段。"""

    state.setdefault("curr_state", "")
    state.setdefault("anonymized_question", "")
    state.setdefault("query_to_retrieve_or_answer", "")
    state.setdefault("plan", [])
    state.setdefault("past_steps", [])
    state.setdefault("mapping", {})
    state.setdefault("curr_context", "")
    state.setdefault("aggregated_context", "")
    state.setdefault("tool", "")
    state.setdefault("response", "")
    return state


def _get_components() -> Dict[str, Any]:
    """懒加载并缓存所有 retriever、链和子工作流组件。"""

    global _COMPONENTS
    if _COMPONENTS is not None:
        return _COMPONENTS

    (
        chunks_query_retriever,
        chapter_summaries_query_retriever,
        book_quotes_query_retriever,
    ) = create_retrievers()

    keep_only_relevant_content_chain = create_keep_only_relevant_content_chain()
    is_distilled_content_grounded_on_content_chain = (
        create_is_distilled_content_grounded_on_content_chain()
    )
    answer_question_from_context_chain = create_answer_question_from_context_chain()
    is_grounded_on_facts_chain = create_is_grounded_on_facts_chain()

    qualitative_chunks_retrieval_workflow_app = (
        create_qualitative_retrieval_book_chunks_workflow_app(
            chunks_query_retriever,
            keep_only_relevant_content_chain,
            is_distilled_content_grounded_on_content_chain,
        )
    )

    qualitative_summaries_retrieval_workflow_app = (
        create_qualitative_summaries_retrieval_workflow_app(
            chapter_summaries_query_retriever,
            keep_only_relevant_content_chain,
            is_distilled_content_grounded_on_content_chain,
        )
    )

    qualitative_book_quotes_retrieval_workflow_app = (
        create_qualitative_book_quotes_retrieval_workflow_app(
            book_quotes_query_retriever,
            keep_only_relevant_content_chain,
            is_distilled_content_grounded_on_content_chain,
        )
    )

    qualitative_answer_workflow_app = create_qualitative_answer_workflow_app(
        answer_question_from_context_chain,
        is_grounded_on_facts_chain,
    )

    _COMPONENTS = {
        "anonymize_question_chain": create_anonymize_question_chain(),
        "planner": create_plan_chain(),
        "de_anonymize_plan_chain": create_deanonymize_plan_chain(),
        "break_down_plan_chain": create_break_down_plan_chain(),
        "task_handler_chain": create_task_handler_chain(),
        "qualitative_chunks_retrieval_workflow_app": qualitative_chunks_retrieval_workflow_app,
        "qualitative_summaries_retrieval_workflow_app": qualitative_summaries_retrieval_workflow_app,
        "qualitative_book_quotes_retrieval_workflow_app": qualitative_book_quotes_retrieval_workflow_app,
        "qualitative_answer_workflow_app": qualitative_answer_workflow_app,
        "replanner": create_replanner_chain(),
        "can_be_answered_already_chain": create_can_be_answered_already_chain(),
    }

    return _COMPONENTS


def _make_keep_only_relevant_content(chain):
    """生成子工作流节点,用于筛选检索结果中的相关内容。"""

    def keep_only_relevant_content(state: QualitativeRetrievalGraphState):
        """根据当前问题从检索上下文中抽取相关内容。"""

        question = state.get("question", "")
        context = state.get("context", "")
        output = chain.invoke({"query": question, "retrieved_documents": context})
        relevant_content = _get_value(output, "relevant_content", "")
        relevant_content = escape_quotes(str(relevant_content))
        return {
            "relevant_context": relevant_content,
            "context": context,
            "question": question,
        }

    return keep_only_relevant_content


def _make_is_distilled_content_grounded_on_content(chain):
    """生成条件判断节点,用于校验提炼内容是否基于原始上下文。"""

    def is_distilled_content_grounded_on_content(state: QualitativeRetrievalGraphState):
        """返回 LangGraph 条件边使用的 grounded 判断结果。"""

        distilled_content = state.get("relevant_context", "")
        original_context = state.get("context", "")
        if not distilled_content:
            return "基于原始语境"
        output = chain.invoke(
            {
                "distilled_content": distilled_content,
                "original_context": original_context,
            }
        )
        grounded = bool(_get_value(output, "grounded", False))
        return "基于原始语境" if grounded else "不基于原始语境"

    return is_distilled_content_grounded_on_content


def create_qualitative_retrieval_book_chunks_workflow_app(
    chunks_query_retriever,
    keep_only_relevant_content_chain,
    is_distilled_content_grounded_on_content_chain,
):
    """创建正文 chunks 检索子工作流。"""

    def retrieve_chunks_context_per_question(state: QualitativeRetrievalGraphState):
        """根据问题检索正文 chunks 并拼接为上下文。"""

        question = state.get("question", "")
        docs = _retrieve_docs(chunks_query_retriever, question)
        context = " ".join(getattr(doc, "page_content", str(doc)) for doc in docs)
        return {"context": escape_quotes(context), "question": question}

    workflow = StateGraph(QualitativeRetrievalGraphState)
    workflow.add_node(
        "retrieve_chunks_context_per_question", retrieve_chunks_context_per_question
    )
    workflow.add_node(
        "keep_only_relevant_content",
        _make_keep_only_relevant_content(keep_only_relevant_content_chain),
    )
    workflow.set_entry_point("retrieve_chunks_context_per_question")
    workflow.add_edge(
        "retrieve_chunks_context_per_question", "keep_only_relevant_content"
    )
    workflow.add_conditional_edges(
        "keep_only_relevant_content",
        _make_is_distilled_content_grounded_on_content(
            is_distilled_content_grounded_on_content_chain
        ),
        {"基于原始语境": END, "不基于原始语境": "keep_only_relevant_content"},
    )
    return workflow.compile()


def create_qualitative_summaries_retrieval_workflow_app(
    chapter_summaries_query_retriever,
    keep_only_relevant_content_chain,
    is_distilled_content_grounded_on_content_chain,
):
    """创建章节摘要检索子工作流。"""

    def retrieve_summaries_context_per_question(state: QualitativeRetrievalGraphState):
        """根据问题检索章节摘要并附带章节信息。"""

        question = state.get("question", "")
        docs = _retrieve_docs(chapter_summaries_query_retriever, question)
        context = " ".join(
            f"{getattr(doc, 'page_content', str(doc))} (chapter {getattr(doc, 'metadata', {}).get('chapter', '')})"
            for doc in docs
        )
        return {"context": escape_quotes(context), "question": question}

    workflow = StateGraph(QualitativeRetrievalGraphState)
    workflow.add_node(
        "retrieve_summaries_context_per_question",
        retrieve_summaries_context_per_question,
    )
    workflow.add_node(
        "keep_only_relevant_content",
        _make_keep_only_relevant_content(keep_only_relevant_content_chain),
    )
    workflow.set_entry_point("retrieve_summaries_context_per_question")
    workflow.add_edge(
        "retrieve_summaries_context_per_question", "keep_only_relevant_content"
    )
    workflow.add_conditional_edges(
        "keep_only_relevant_content",
        _make_is_distilled_content_grounded_on_content(
            is_distilled_content_grounded_on_content_chain
        ),
        {"基于原始语境": END, "不基于原始语境": "keep_only_relevant_content"},
    )
    return workflow.compile()


def create_qualitative_book_quotes_retrieval_workflow_app(
    book_quotes_query_retriever,
    keep_only_relevant_content_chain,
    is_distilled_content_grounded_on_content_chain,
):
    """创建书籍引文检索子工作流。"""

    def retrieve_book_quotes_context_per_question(
        state: QualitativeRetrievalGraphState,
    ):
        """根据问题检索相关原文引文并拼接上下文。"""

        question = state.get("question", "")
        docs = _retrieve_docs(book_quotes_query_retriever, question)
        context = " ".join(getattr(doc, "page_content", str(doc)) for doc in docs)
        return {"context": escape_quotes(context), "question": question}

    workflow = StateGraph(QualitativeRetrievalGraphState)
    workflow.add_node(
        "retrieve_book_quotes_context_per_question",
        retrieve_book_quotes_context_per_question,
    )
    workflow.add_node(
        "keep_only_relevant_content",
        _make_keep_only_relevant_content(keep_only_relevant_content_chain),
    )
    workflow.set_entry_point("retrieve_book_quotes_context_per_question")
    workflow.add_edge(
        "retrieve_book_quotes_context_per_question", "keep_only_relevant_content"
    )
    workflow.add_conditional_edges(
        "keep_only_relevant_content",
        _make_is_distilled_content_grounded_on_content(
            is_distilled_content_grounded_on_content_chain
        ),
        {"基于原始语境": END, "不基于原始语境": "keep_only_relevant_content"},
    )
    return workflow.compile()


def create_qualitative_answer_workflow_app(
    answer_question_from_context_chain,
    is_grounded_on_facts_chain,
):
    """创建基于上下文回答并校验事实性的子工作流。"""

    class QualitativeAnswerGraphState(TypedDict, total=False):
        question: str
        context: str
        answer: str

    def answer_question_from_context(state: QualitativeAnswerGraphState):
        """调用问答链生成候选答案。"""

        question = state.get("question", "")
        context = state.get("context", "")
        output = answer_question_from_context_chain.invoke(
            {"question": question, "context": context}
        )
        answer = _get_value(output, "answer", "根据给定上下文无法确定。")
        return {"question": question, "context": context, "answer": str(answer)}

    def is_answer_grounded_on_context(state: QualitativeAnswerGraphState):
        """判断候选答案是否完全基于上下文。"""

        context = state.get("context", "")
        answer = state.get("answer", "")
        output = is_grounded_on_facts_chain.invoke(
            {"context": context, "answer": answer}
        )
        grounded = bool(_get_value(output, "grounded_on_facts", True))
        return "基于上下文" if grounded else "幻觉"

    workflow = StateGraph(QualitativeAnswerGraphState)
    workflow.add_node("answer_question_from_context", answer_question_from_context)
    workflow.set_entry_point("answer_question_from_context")
    workflow.add_conditional_edges(
        "answer_question_from_context",
        is_answer_grounded_on_context,
        {"幻觉": "answer_question_from_context", "基于上下文": END},
    )
    return workflow.compile()


def anonymize_queries(state: PlanExecute):
    """LangGraph 节点: 对用户问题做匿名化处理。"""

    state = _init_state(state)
    state["curr_state"] = "anonymize_question"
    chain = _get_components()["anonymize_question_chain"]
    output = chain.invoke({"question": state["question"]})
    state["anonymized_question"] = str(
        _get_value(output, "anonymized_question", state["question"])
    )
    mapping = _get_value(output, "mapping", {})
    state["mapping"] = mapping if isinstance(mapping, dict) else {}
    return state


def plan_step(state: PlanExecute):
    """LangGraph 节点: 根据匿名化问题生成初始计划。"""

    state["curr_state"] = "planner"
    planner = _get_components()["planner"]
    plan_output = planner.invoke(
        {"question": state.get("anonymized_question") or state["question"]}
    )
    steps = _get_steps(plan_output)
    if not steps:
        steps = [f"检索与问题相关的信息：{state['question']}"]
    state["plan"] = steps
    return state


def deanonymize_queries(state: PlanExecute):
    """LangGraph 节点: 将计划中的变量还原为原始实体。"""

    state["curr_state"] = "de_anonymize_plan"
    chain = _get_components()["de_anonymize_plan_chain"]
    output = chain.invoke(
        {"plan": state.get("plan", []), "mapping": state.get("mapping", {})}
    )
    steps = _get_steps(_get_value(output, "plan", output))
    state["plan"] = steps or state.get("plan", [])
    return state


def break_down_plan_step(state: PlanExecute):
    """LangGraph 节点: 把计划拆成检索或回答任务。"""

    state["curr_state"] = "break_down_plan"
    if not state.get("plan"):
        if state.get("aggregated_context"):
            state["plan"] = [f"根据已有上下文回答问题：{state['question']}"]
        else:
            state["plan"] = [f"检索与问题相关的信息：{state['question']}"]

    chain = _get_components()["break_down_plan_chain"]
    output = chain.invoke({"plan": state.get("plan", [])})
    steps = _get_steps(output)
    state["plan"] = steps or state.get("plan", [])
    return state


def run_task_handler_chain(state: PlanExecute):
    """LangGraph 节点: 选择下一步要调用的工具并准备输入。"""

    state["curr_state"] = "task_handler"
    state.setdefault("past_steps", [])

    if not state.get("plan"):
        if state.get("aggregated_context"):
            state["query_to_retrieve_or_answer"] = state["question"]
            state["curr_context"] = state.get("aggregated_context", "")
            state["tool"] = "answer"
            return state
        state["plan"] = [f"检索与问题相关的信息：{state['question']}"]

    curr_task = state["plan"][0]
    chain = _get_components()["task_handler_chain"]
    output = chain.invoke(
        {
            "curr_task": curr_task,
            "aggregated_context": state.get("aggregated_context", ""),
            "last_tool": state.get("tool", ""),
            "past_steps": state.get("past_steps", []),
            "question": state["question"],
        }
    )

    state["past_steps"].append(curr_task)
    state["plan"].pop(0)

    tool = _normalize_tool_name(str(_get_value(output, "tool", "")))
    query = str(_get_value(output, "query", curr_task))
    curr_context = str(_get_value(output, "curr_context", ""))

    if tool == "retrieve_chunks":
        state["query_to_retrieve_or_answer"] = query
        state["tool"] = "retrieve_chunks"
    elif tool == "retrieve_summaries":
        state["query_to_retrieve_or_answer"] = query
        state["tool"] = "retrieve_summaries"
    elif tool == "retrieve_quotes":
        state["query_to_retrieve_or_answer"] = query
        state["tool"] = "retrieve_quotes"
    elif tool == "answer_from_context":
        state["query_to_retrieve_or_answer"] = query
        state["curr_context"] = curr_context or state.get("aggregated_context", "")
        state["tool"] = "answer"
    else:
        # 模型偶尔会输出不规范工具名,做一个稳妥兜底。
        state["query_to_retrieve_or_answer"] = query or state["question"]
        state["tool"] = "retrieve_chunks"

    return state


def retrieve_or_answer(state: PlanExecute):
    """LangGraph 条件节点: 根据工具选择返回后续分支名。"""

    state["curr_state"] = "decide_tool"
    if state.get("tool") == "retrieve_chunks":
        return "chosen_tool_is_retrieve_chunks"
    if state.get("tool") == "retrieve_summaries":
        return "chosen_tool_is_retrieve_summaries"
    if state.get("tool") == "retrieve_quotes":
        return "chosen_tool_is_retrieve_quotes"
    if state.get("tool") == "answer":
        return "chosen_tool_is_answer"
    return "chosen_tool_is_retrieve_chunks"


def run_qualitative_chunks_retrieval_workflow(state: PlanExecute):
    """LangGraph 节点: 运行正文 chunks 检索工作流并追加上下文。"""

    state["curr_state"] = "retrieve_chunks"
    app = _get_components()["qualitative_chunks_retrieval_workflow_app"]
    result = _run_sub_workflow_and_get_last_state(
        app,
        {"question": state.get("query_to_retrieve_or_answer", state["question"])},
    )
    return _append_context(state, result.get("relevant_context", ""))


def run_qualitative_summaries_retrieval_workflow(state: PlanExecute):
    """LangGraph 节点: 运行章节摘要检索工作流并追加上下文。"""

    state["curr_state"] = "retrieve_summaries"
    app = _get_components()["qualitative_summaries_retrieval_workflow_app"]
    result = _run_sub_workflow_and_get_last_state(
        app,
        {"question": state.get("query_to_retrieve_or_answer", state["question"])},
    )
    return _append_context(state, result.get("relevant_context", ""))


def run_qualitative_book_quotes_retrieval_workflow(state: PlanExecute):
    """LangGraph 节点: 运行书籍引文检索工作流并追加上下文。"""

    state["curr_state"] = "retrieve_book_quotes"
    app = _get_components()["qualitative_book_quotes_retrieval_workflow_app"]
    result = _run_sub_workflow_and_get_last_state(
        app,
        {"question": state.get("query_to_retrieve_or_answer", state["question"])},
    )
    return _append_context(state, result.get("relevant_context", ""))


def run_qualtative_answer_workflow(state: PlanExecute):
    """LangGraph 节点: 基于当前上下文生成中间答案并追加到上下文。"""

    state["curr_state"] = "answer"
    app = _get_components()["qualitative_answer_workflow_app"]
    result = _run_sub_workflow_and_get_last_state(
        app,
        {
            "question": state.get("query_to_retrieve_or_answer", state["question"]),
            "context": state.get("curr_context") or state.get("aggregated_context", ""),
        },
    )
    return _append_context(state, result.get("answer", ""))


def replan_step(state: PlanExecute):
    """LangGraph 节点: 根据已获得上下文决定是否还要继续规划。"""

    state["curr_state"] = "replan"

    if state.get("aggregated_context") and not state.get("plan"):
        return state

    chain = _get_components()["replanner"]
    output = chain.invoke(
        {
            "question": state["question"],
            "plan": state.get("plan", []),
            "past_steps": state.get("past_steps", []),
            "aggregated_context": state.get("aggregated_context", ""),
        }
    )
    state["plan"] = _get_steps(output)
    return state


def run_qualtative_answer_workflow_for_final_answer(state: PlanExecute):
    """LangGraph 节点: 基于聚合上下文生成最终答案。"""

    state["curr_state"] = "get_final_answer"
    app = _get_components()["qualitative_answer_workflow_app"]
    result = _run_sub_workflow_and_get_last_state(
        app,
        {"question": state["question"], "context": state.get("aggregated_context", "")},
    )
    state["response"] = str(result.get("answer", "根据给定上下文无法确定。"))
    return state


def can_be_answered(state: PlanExecute):
    """LangGraph 条件节点: 判断当前上下文是否已经足够回答。"""

    state["curr_state"] = "can_be_answered_already"
    context = state.get("aggregated_context", "")
    plan = state.get("plan", [])

    if context and not plan:
        return "can_be_answered_already"

    if not context and not plan:
        state["plan"] = [f"检索与问题相关的信息：{state['question']}"]
        return "cannot_be_answered_yet"

    chain = _get_components()["can_be_answered_already_chain"]
    output = chain.invoke({"question": state["question"], "context": context})
    if bool(_get_value(output, "can_be_answered", False)):
        return "can_be_answered_already"

    if not plan:
        return "can_be_answered_already" if context else "cannot_be_answered_yet"
    return "cannot_be_answered_yet"


def create_agent():
    """创建完整的计划执行 Agent 工作流并编译为可运行应用。"""

    _get_components()

    agent_workflow = StateGraph(PlanExecute)

    agent_workflow.add_node("anonymize_question", anonymize_queries)
    agent_workflow.add_node("planner", plan_step)
    agent_workflow.add_node("de_anonymize_plan", deanonymize_queries)
    agent_workflow.add_node("break_down_plan", break_down_plan_step)
    agent_workflow.add_node("task_handler", run_task_handler_chain)
    agent_workflow.add_node(
        "retrieve_chunks", run_qualitative_chunks_retrieval_workflow
    )
    agent_workflow.add_node(
        "retrieve_summaries", run_qualitative_summaries_retrieval_workflow
    )
    agent_workflow.add_node(
        "retrieve_book_quotes", run_qualitative_book_quotes_retrieval_workflow
    )
    agent_workflow.add_node("answer", run_qualtative_answer_workflow)
    agent_workflow.add_node("replan", replan_step)
    agent_workflow.add_node(
        "get_final_answer", run_qualtative_answer_workflow_for_final_answer
    )

    agent_workflow.set_entry_point("anonymize_question")

    agent_workflow.add_edge("anonymize_question", "planner")
    agent_workflow.add_edge("planner", "de_anonymize_plan")
    agent_workflow.add_edge("de_anonymize_plan", "break_down_plan")
    agent_workflow.add_edge("break_down_plan", "task_handler")

    agent_workflow.add_conditional_edges(
        "task_handler",
        retrieve_or_answer,
        {
            "chosen_tool_is_retrieve_chunks": "retrieve_chunks",
            "chosen_tool_is_retrieve_summaries": "retrieve_summaries",
            "chosen_tool_is_retrieve_quotes": "retrieve_book_quotes",
            "chosen_tool_is_answer": "answer",
        },
    )

    agent_workflow.add_edge("retrieve_chunks", "replan")
    agent_workflow.add_edge("retrieve_summaries", "replan")
    agent_workflow.add_edge("retrieve_book_quotes", "replan")
    agent_workflow.add_edge("answer", "replan")

    agent_workflow.add_conditional_edges(
        "replan",
        can_be_answered,
        {
            "can_be_answered_already": "get_final_answer",
            "cannot_be_answered_yet": "break_down_plan",
        },
    )

    agent_workflow.add_edge("get_final_answer", END)

    return agent_workflow.compile()
