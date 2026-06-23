from importlib import metadata
from pydoc import describe, doc
from re import search, template
from typing import List, TypedDict

from fsspec import mapping
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph
from pydantic import Field
from pydantic import BaseModel
from langgraph.graph import END, StateGraph
from langchain.vectorstores import FAISS

from functions_for_pipeline import answer_question_from_context
from helper_functions import escape_quotes
from sophisticated_rag_agent_harry_potter import (
    is_grounded_on_facts_prompt,
    is_grounded_on_facts_prompt_template,
    qualitative_answer_workflow,
    question,
)


class PlanExecute(TypedDict):
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
    """未来要执行的计划"""

    steps: List[str] = Field(description="要遵循的不同步骤,应按顺序排列")


def create_retrievers():
    embeddings = OpenAIEmbeddings()
    chunks_vector_store = FAISS.load_local(
        "chunks_vector_store", embeddings, allow_dangerous_deserialization=True
    )
    chapter_summaries_vector_store = FAISS.load_local(
        "chapter_summaries_vector_store",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    book_quotes_vectorstore = FAISS.load_local(
        "book_quotes_vectorstore", embeddings, allow_dangerous_deserialization=True
    )

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


(
    chunks_query_retriever,
    chapter_summaries_query_retriever,
    book_quotes_query_retriever,
) = create_retrievers()


def create_plan_chain():
    planner_prompt = """对于给定的查询{question},请提供一个简单的分步计划,说明如何得出答案。
    该计划应包含各项具体任务,若执行得当,将得出正确答案。请勿添加任何多余步骤。 
    最后一步的结果应该是最终答案。确保每一步都包含了所有必要的信息——不要跳过任何步骤。
    """

    planner_prompt = PromptTemplate(
        template=planner_prompt, input_variables=["question"]
    )
    planner_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
    planner = planner_prompt | planner_llm.with_structured_output(Plan)
    return planner


def create_anonymize_question_chain():
    class AnonymizeQuestion(BaseModel):
        """匿名化问题及其映射关系"""

        anonymized_question: str = Field(description="匿名问题")
        mapping: dict = Field(description="将原始名称实体映射到变量。")
        explanation: str = Field(description="动作说明")

    anonymize_question_parser = JsonOutputParser(pydantic_object=AnonymizeQuestion)
    anonymize_question_prompt_template = """你是一个问题匿名化工具。你接收到的输入是一个包含多个单词的字符串
    构造一个问题{question}。你的目标是将输入中的所有名称实体替换为变量,并记住原始名称实体到变量的映射关系。
    ```example1:
            如果输入是“谁是谁哈利波特?”,输出应该是“谁是谁X?”,映射应该是{{“X”: “哈利波特”}}
    ```example2:
            如果输入是“那个坏人跟亚历克斯和罗尼是怎么玩的?”
            输出应该是“X是如何与Y和Z一起玩的?”并且映射应该是{{"X":"坏人","Y":"亚历克斯","Z":"罗尼"}}
    你必须将输入中的所有命名实体替换为变量,并记住原始命名实体到变量的映射关系。
    按照此处所述的json格式,将匿名化问题和映射作为两个单独的字段输出,除json格式外,不包含任何其他文本。
   """

    anonymize_question_prompt = PromptTemplate(
        template=anonymize_question_prompt_template,
        input_variables=["question"],
        partial_variables={
            "format_instructions": anonymize_question_parser.get_format_instructions()
        },
    )

    anonymize_question_llm = ChatOpenAI(
        temperature=0, model_name="gpt-4o", max_tokens=2000
    )
    anonymize_question_chain = (
        anonymize_question_prompt | anonymize_question_llm | anonymize_question_parser
    )
    return anonymize_question_chain


def create_deanonymize_plan_chain():
    class DeAnonymizePlan(BaseModel):
        """该动作可能产生的结果"""

        plan: List = Field(description="计划将来跟进。所有变量都替换为映射的单词。")

    de_anonymize_plan_prompt_template = """你收到一个任务列表:{plan},其中一些单词被替换为映射的变量。你还收到了
    将这些变量映射到单词{mapping}。用映射后的单词替换任务列表中的所有变量。如果不存在变量,
    返回原始任务列表。在任何情况下,只需按照此处所述的json格式输出更新后的任务列表,除“[原文内容]”外,不添加任何其他文本"""

    de_anonymize_plan_prompt = PromptTemplate(
        template=de_anonymize_plan_prompt_template, input_variables=["plan", "mapping"]
    )

    de_anonymize_plan_llm = ChatOpenAI(
        temperature=0, model_name="gpt-4o", max_tokens=2000
    )
    de_anonymize_plan_chain = (
        de_anonymize_plan_prompt
        | de_anonymize_plan_llm.with_structured_output(DeAnonymizePlan)
    )
    return de_anonymize_plan_chain


def create_break_down_plan_chain():
    break_down_plan_prompt_template = """你收到一个计划{plan},其中包含了一系列为回答某个问题而需遵循的步骤。
    你需要仔细审查这个计划,并根据以下内容进行完善:
    1. 每一步都必须能够由以下任一方式执行:
        i. 从书籍片段的向量存储中检索相关信息
        ii. 从章节摘要的向量存储中检索相关信息
        iii. 从书籍引文的向量存储中检索相关信息
        iv. 根据给定上下文回答问题。
    2. 每一步都应包含执行该步骤所需的所有信息。
    输出优化后的计划"""

    break_down_plan_prompt = PromptTemplate(
        template=break_down_plan_prompt_template, input_variables=["plan"]
    )

    break_down_plan_llm = ChatOpenAI(
        temperature=0, model_name="gpt-4o", max_tokens=2000
    )

    break_down_plan_chain = (
        break_down_plan_prompt | break_down_plan_llm.with_structured_output(Plan)
    )

    return break_down_plan_chain


def create_task_handler_chain():
    tasks_handler_prompt_template = """你是一名任务处理者,接收到任务{curr_task}后,必须决定使用哪种工具来执行该任务。
    您手头有以下工具可供使用:
    工具A:一种根据给定查询从书籍片段的向量存储中检索相关信息的工具。
    - 当你认为当前任务应在书籍章节中搜索信息时,使用工具A。
    Took B:一种工具,可根据给定查询从章节摘要的向量存储中检索相关信息。
    - 当你认为当前任务应在章节摘要中搜索信息时,请使用工具B。
    工具C:一种根据给定查询从书籍报价的向量存储中检索相关信息的工具。
    - 当你认为当前任务应在书籍引文中搜索信息时,请使用工具C。
    工具D:一种能从给定上下文中回答问题的工具。
    - 仅当当前任务可以通过聚合上下文{aggregated_context}来回答时,才使用工具D
    你还会收到最后使用的工具{last_tool}
    如果{last_tool}是retrieve_chunks,则使用除工具A以外的其他工具。
    你还可以利用过去的步骤{past_steps}来做出决策,并理解任务的上下文。
    你还可以使用初始用户的问题{question}来做出决策并理解任务的上下文。
    如果你决定使用工具A、B或C,请输出针对该工具的查询语句,并同时输出相关的工具名称。
    如果你决定使用工具D,请输出该工具所对应的问题、上下文,并明确指出所使用的工具是工具D。
    """

    class TaskHandlerOutput(BaseModel):
        """任务处理器的输出结构"""

        query: str = Field(
            description="查询内容要么从向量存储中检索,要么是从上下文中应回答的问题。"
        )
        curr_context: str = Field(description="为回答该问题而需依据的上下文。")
        tool: str = Field(
            description="所使用的工具应为retrieve_chunks、retrieve_summaries、retrieve_quotes或answer_from_context中的任何一个。"
        )

    task_handler_prompt = PromptTemplate(
        template=tasks_handler_prompt_template,
        input_variables=[
            "curr_task",
            "aggregated_context",
            "last_tool",
            "past_steps",
            "question",
        ],
    )

    task_handler_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
    task_handler_chain = task_handler_prompt | task_handler_llm.with_structured_output(
        TaskHandlerOutput
    )

    return task_handler_chain


class QualitativeRetrievalGraphState(TypedDict):
    """表示当前图工作流的状态结构"""

    question: str
    context: str
    relevant_context: str


def retrieve_chunks_context_per_question(state):
    """为给定问题检索相关上下文,上下文来自书籍分块与章节摘要"""
    question = state["question"]
    docs = chunks_query_retriever.get_relevant_documents(question)

    # 拼接文档内容
    context = " ".join(doc.page_content for doc in docs)
    context = escape_quotes(context)

    return {"context": context, "question": question}


def create_keep_only_relevant_content_chain():
    keep_only_relevant_content_prompt_template = """您收到一个查询:{query},并从a中检索到文档:{retrieved_documents}”
    向量存储。
    你需要过滤掉所有与{query}无关且无法提供重要信息的非相关信息。
    你的目标只是过滤掉不相关的信息。
    你可以删除句子中与查询无关的部分,或者删除与查询无关的整个句子。
    请勿添加任何未包含在检索到的文件中的新信息。
    输出经过过滤的相关内容。"""

    class KeepRelevantContent(BaseModel):
        relevant_content: str = Field(
            description="从检索到的文档中提取的与查询相关的内容。"
        )

    keep_only_relevant_content_prompt = PromptTemplate(
        template=keep_only_relevant_content_prompt_template,
        input_variables=["query", "retrieved_documents"],
    )

    keep_only_relevant_content_llm = ChatOpenAI(
        temperature=0, model_name="gpt-4o", max_tokens=2000
    )

    keep_only_relevant_content_chain = (
        keep_only_relevant_content_prompt
        | keep_only_relevant_content_llm.with_structured_output(KeepRelevantContent)
    )

    return keep_only_relevant_content_chain


keep_only_relevant_content_chain = create_keep_only_relevant_content_chain()


def keep_only_relevant_content(state):
    """仅保留检索文档中与查询相关的内容"""
    question = state["question"]
    context = state["context"]

    input_data = {"query": question, "retrieved_documents": context}
    output = keep_only_relevant_content_chain.invoke(input_data)

    relevant_content = output.relevant_content
    relevant_content = "".join(relevant_content)
    relevant_content = escape_quotes(relevant_content)

    return {
        "relevant_context": relevant_content,
        "context": context,
        "question": question,
    }


def create_is_distilled_content_grounded_on_content_chain():
    is_distilled_content_grounded_on_content_prompt_template = """您将收到一些提炼的内容：{distilled_content} 以及原始上下文：{original_context}。
        你需要判断提炼的内容是否基于原始上下文。
        如果提取的内容基于原始上下文,则将基于上下文字段设置为true。
        如果提取的内容未基于原始上下文,则将基于上下文字段设置为false。"""

    class IsDistilledContentGroundedOnContent(BaseModel):
        grounded: bool = Field(description="提炼的内容是否基于原始上下文。")
        explanation: str = Field(
            description="解释为何提炼的内容基于或未基于原始上下文。"
        )

    is_distilled_content_grounded_on_content_prompt = PromptTemplate(
        template=is_distilled_content_grounded_on_content_prompt_template,
        input_variables=["distilled_content", "original_context"],
    )

    is_distilled_content_grounded_on_content_llm = ChatOpenAI(
        temperature=0, model_name="gpt-4o", max_tokens=2000
    )

    is_distilled_content_grounded_on_content_chain = (
        is_distilled_content_grounded_on_content_prompt
        | is_distilled_content_grounded_on_content_llm.with_structured_output(
            IsDistilledContentGroundedOnContent
        )
    )

    return is_distilled_content_grounded_on_content_chain


is_distilled_content_grounded_on_content_chain = (
    create_is_distilled_content_grounded_on_content_chain()
)


def is_distilled_content_grounded_on_content(state):
    """
    判断提炼后的内容是否基于原始上下文

    Args:
    distilled_content:提炼后的内容.
    original_context:原始上下文.
    """
    distilled_content = state["relevant_context"]
    original_context = state["context"]

    input_data = {
        "distilled_content": distilled_content,
        "original_context": original_context,
    }

    output = is_distilled_content_grounded_on_content_chain.invoke(input_data)
    grounded = output.grounded

    if grounded:
        return "基于原始语境"
    else:
        return "不基于原始语境"


def create_qualitative_retrieval_book_chunks_workflow_app():
    """构建并编译一个用于“定性检索书本切片(chunks)”的工作流应用"""

    # 初始化工作流图
    qualitative_chunks_retrieval_workflow = StateGraph(QualitativeRetrievalGraphState)
    # 根据问题检索书本的切片上下文
    qualitative_chunks_retrieval_workflow.add_node(
        "retrieve_chunks_context_per_question", retrieve_chunks_context_per_question
    )
    # 仅保留相关内容
    qualitative_chunks_retrieval_workflow.add_node(
        "keep_only_relevant_content", keep_only_relevant_content
    )

    # 构建图
    qualitative_chunks_retrieval_workflow.set_entry_point(
        "retrieve_chunks_context_per_question"
    )

    qualitative_chunks_retrieval_workflow.add_edge(
        "retrieve_chunks_context_per_question", "keep_only_relevant_content"
    )

    # 判断执行完后的内容是否忠实于原始语境,即检查LLM有没有产生幻觉
    qualitative_chunks_retrieval_workflow.add_conditional_edges(
        "keep_only_relevant_content",
        is_distilled_content_grounded_on_content,
        {"基于原始语境": END, "没有基于原始语境": "keep_only_relevant_content"},
    )

    # 将点、线、逻辑编译成一个可执行的LangGraph应用程序
    qualitative_chunks_retrieval_workflow_app = (
        qualitative_chunks_retrieval_workflow.compile()
    )

    return qualitative_chunks_retrieval_workflow_app


def retrieve_summaries_context_per_question(state):
    """按问题检索摘要上下文"""
    question = state["question"]
    docs_summaries = chapter_summaries_query_retriever.get_relevant_documents(
        state["question"]
    )

    # 拼接章节摘要并附带引用信息
    context_summaries = " ".join(
        f"{doc.page_conte} (chapter {doc.metadata['chapter']})"
        for doc in docs_summaries
    )
    context_summaries = escape_quotes(context_summaries)
    return {"context": context_summaries, "question": question}


def create_qualitative_summaries_retrieval_workflow_app():
    qualitative_summaries_retrieval_workflow = StateGraph(
        QualitativeRetrievalGraphState
    )
    # 定义节点
    qualitative_summaries_retrieval_workflow.add_node(
        "retrieve_summaries_context_per_question",
        retrieve_summaries_context_per_question,
    )
    qualitative_summaries_retrieval_workflow.add_node(
        "keep_only_relevant_content", keep_only_relevant_content
    )

    # 构建图
    qualitative_summaries_retrieval_workflow.set_entry_point(
        "retrieve_summaries_context_per_question"
    )

    qualitative_summaries_retrieval_workflow.add_edge(
        "retrieve_summaries_context_per_question", "keep_only_relevant_content"
    )
    qualitative_summaries_retrieval_workflow.add_conditional_edges(
        "keep_only_relevant_content",
        is_distilled_content_grounded_on_content,
        {"基于原始语境": END, "不基于原始语境": "keep_only_relevant_content"},
    )


def retrieve_book_quotes_context_per_question(state):
    """检索每个问题的书籍报价上下文"""
    question = state["question"]
    docs_book_quotes = book_quotes_query_retriever.get_relevant_documents(
        state["question"]
    )
    book_quotes = " ".join(doc.page_content for doc in docs_book_quotes)
    book_quotes_content = escape_quotes(book_quotes)

    return {"context": book_quotes_content, "question": question}


def create_qualitative_book_quotes_retrieval_workflow_app():
    qualitative_book_quotes_retrieval_workflow = StateGraph(
        QualitativeRetrievalGraphState
    )

    # 定义节点
    qualitative_book_quotes_retrieval_workflow.add_node(
        "retrieve_book_quotes_context_per_question",
        retrieve_book_quotes_context_per_question,
    )
    qualitative_book_quotes_retrieval_workflow.add_node(
        "keep_only_relevant_content", keep_only_relevant_content
    )

    # 构建图
    qualitative_book_quotes_retrieval_workflow.set_entry_point(
        "retrieve_book_quotes_context_per_question"
    )

    qualitative_book_quotes_retrieval_workflow.add_edge(
        "retrieve_book_quotes_context_per_question", "keep_only_relevant_content"
    )

    qualitative_book_quotes_retrieval_workflow.add_conditional_edges(
        "keep_only_relevant_content",
        is_distilled_content_grounded_on_content,
        {"基于原始语境": END, "不基于原始语境": "keep_only_relevant_content"},
    )


def create_is_grounded_on_facts_chain():
    class is_grounded_on_facts(BaseModel):
        # 重写问题的输出结构定义
        grounded_on_facts: bool = Field(description="答案基于事实,“是”或“否”")

    is_grounded_on_facts_llm = ChatOpenAI(
        temperature=0, model_name="gpt-4o", max_tokens=2000
    )
    is_grounded_on_facts_prompt_template = """你是一名事实核查员,负责判断给定答案{answer}是否基于给定上下文{context}
    只要它基于上下文,即使没有意义,你也不介意。
    输出一个包含问题答案的JSON格式数据,除了JSON格式的数据外,不要输出任何额外的文本。
    """

    is_grounded_on_facts_prompt = PromptTemplate(
        template=is_grounded_on_facts_prompt_template,
        input_variables=["context", "answer"],
    )

    is_grounded_on_facts_chain = (
        is_grounded_on_facts_prompt
        | is_grounded_on_facts_llm.with_structured_output(is_grounded_on_facts)
    )

    return is_grounded_on_facts_chain


is_grounded_on_facts_chain = create_is_grounded_on_facts_chain()


def is_answer_grounded_on_context(state):
    # 判断问题答案是否基于事实
    context = state["context"]
    answer = state["answer"]

    result = is_grounded_on_facts_chain.invoke({"context": context, "answer": answer})

    grounded_on_facts = result.grounded_on_facts

    if not grounded_on_facts:
        return "幻觉"
    else:
        return "基于上下文"


def create_qualitative_answer_workflow_app():
    class QualitativeAnswerGraphState(TypedDict):
        # 表示当前图工作流的状态结构
        question: str
        context: str
        answer: str

    qualitative_answer_workflow = StateGraph(QualitativeAnswerGraphState)

    # 定义节点

    qualitative_answer_workflow.add_node(
        "answer_question_from_context", answer_question_from_context
    )

    # 构建图
    qualitative_answer_workflow.set_entry_point("answer_question_from_context")

    qualitative_answer_workflow.add_conditional_edges(
        "answer_question_from_context",
        is_answer_grounded_on_context,
        {"幻觉": "answer_question_from_context", "基于上下文": END},
    )

    qualitative_answer_workflow_app = qualitative_answer_workflow.compile()
    return qualitative_answer_workflow_app


anonymize_question_chain = create_anonymize_question_chain()
planner = create_plan_chain()
de_anonymize_plan_chain = create_deanonymize_plan_chain()
break_down_plan_chain = create_break_down_plan_chain()
task_handler_chain = create_task_handler_chain()
qualitative_chunks_retrieval_workflow_app = (
    create_qualitative_retrieval_book_chunks_workflow_app()
)
qualitative_summaries_retrieval_workflow_app = (
    create_qualitative_summaries_retrieval_workflow_app()
)
qualitative_book_quotes_retrieval_workflow_app = (
    create_qualitative_book_quotes_retrieval_workflow_app()
)
qualitative_answer_workflow_app = create_qualitative_answer_workflow_app()


def anonymize_queries(state: PlanExecute):
    """对问题进行匿名化处理"""
    state["curr_state"] = "anonymize_question"
    input_values = {"question": state["question"]}
    anonymized_question_output = anonymize_question_chain.invoke(input_values)
    anonymized_question = anonymized_question_output["anonymized_question"]
    mapping = anonymized_question_output["mapping"]
    state["anonymized_question"] = anonymized_question
    state["mapping"] = mapping
    return state


def plan_step(state: PlanExecute):
    """规划下一步动作"""
    state["curr_state"] = "planner"
    plan = planner.invoke({"question": state["anonymized_question"]})
    state["plan"] = plan.steps
    return state


def deanonymize_queries(state: PlanExecute):
    """对计划执行去匿名化还原"""
    state["curr_state"] = "de_anonymize_plan"
    deanonimzed_plan = de_anonymize_plan_chain.invoke(
        {"plan": state["plan"], "mapping": state["mapping"]}
    )
    state["plan"] = deanonimzed_plan.plan
    return state


def break_down_plan_step(state: PlanExecute):
    """将计划步骤拆分为可检索或可回答的任务"""
    state["curr_state"] = "break_down_plan"
    refined_plan = break_down_plan_chain.invoke(state["plan"])
    state["plan"] = refined_plan.steps
    return state


def run_task_handler_chain(state: PlanExecute):
    "运行任务处理链,决定执行当前任务应使用的工具"
    state["curr_state"] = "task_handler"
    if not state["past_steps"]:
        state["past_steps"] = []
    curr_task = state["plan"][0]

    inputs = {
        "curr_task": curr_task,
        "aggregated_context": state["aggregated_context"],
        "last_tool": state["tool"],
        "past_steps": state["past_steps"],
        "question": state["question"],
    }

    output = task_handler_chain.invoke(inputs)

    state["past_steps"].append(curr_task)
    state["plan"].pop(0)

    if output.tool == "retrieve_chunks":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"] = "retrieve_chunks"
    elif output.tool == "retrieve_summaries":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"] = "retrieve_summaries"
    elif output.tool == "retrieve_quotes":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"] = "retrieve_quotes"
    elif output.tool == "answer_from_context":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"] = "answer"
    else:
        raise ValueError("输出的工具无效。必须是'retrieve'或'answer_from_context'")
    return state


def retrieve_or_answer(state: PlanExecute):
    """基于当前状态决定执行检索还是回答"""
    state["curr_state"] = "decide_tool"
    if state["tool"] == "retrieve_chunks":
        return "chosen_tool_is_retrieve_chunks"
    elif state["tool"] == "retrieve_summaries":
        return "chosen_tool_is_retrieve_summaries"
    elif state["tool"] == "retrieve_quotes":
        return "chosen_tool_is_retrieve_quotes"
    elif state["tool"] == "answer":
        return "chosen_tool_is_answer"
    else:
        raise ValueError("输出的工具无效。必须是“retrieve”或“answer_from_context”")


def run_qualitative_chunks_retrieval_workflow(state):
    """运行定性分块检索工作流"""
    state["curr_state"] = "retrieve_chunks"
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    for output in qualitative_chunks_retrieval_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass
    if not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output["relevant_context"]
    return state


def run_qualitative_summaries_retrieval_workflow(state):
    """运行定性摘要检索工作流"""
    state["curr_state"] = "retrieve_summaries"
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    for output in qualitative_summaries_retrieval_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass
    if not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output["relevant_context"]
    return state


def run_qualitative_book_quotes_retrieval_workflow(state):
    """运行定性书摘检索工作流"""
    state["curr_state"] = "retrieve_book_quotes"
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    for output in qualitative_book_quotes_retrieval_workflow_app.stream(inputs):
        for _, _ in output.item():
            pass
    if not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output["aggregated_context"]
    return state


def run_qualtative_answer_workflow(state):
    """运行定性回答工作流"""
    state["curr_state"] = "answer"
    question = state["query_to_retrieve_or answer"]
    context = state["curr_context"]
    inputs = {"question": question, "context": context}

    for output in qualitative_answer_workflow_app.stream(inputs):
        for _, _ in output.items():
            pass

    if not state["aggregated_context"]:
        state["aggregated_context"] = ""
    state["aggregated_context"] += output["answer"]

    return state


def create_agent():
    agent_workflow = StateGraph(PlanExecute)
    # 添加匿名化节点
    agent_workflow.add_node("anonymize_question", anonymize_queries)
    # 添加规划节点
    agent_workflow.add_node("planner", plan_step)
    # 添加去匿名化节点
    agent_workflow.add_node("de_anonymize_plan", deanonymize_queries)
    # 添加计划拆解节点
    agent_workflow.add_node("break_down_plan", break_down_plan_step)
    # 添加任务处理节点
    agent_workflow.add_node("task_handler", run_task_handler_chain)
    # 添加定性分块检索节点
    agent_workflow.add_node(
        "retrieve_chunks", run_qualitative_chunks_retrieval_workflow
    )
    # 添加定性摘要检查节点
    agent_workflow.add_node(
        "retrieve_summaries", run_qualitative_summaries_retrieval_workflow
    )
    # 添加定性书摘检索节点
    agent_workflow.add_node(
        "retrieve_book_quotes", run_qualitative_book_quotes_retrieval_workflow
    )
    # 添加定性回答节点
    agent_workflow.add_node("answer", run_qualtative_answer_workflow)

    # 设置入口节点
    agent_workflow.set_entry_point("anonymize_question")

    # 从匿名化流转到规划
    agent_workflow.add_edge("anonymize_question", "planner")
    # 从规划流转到去匿名化
    agent_workflow.add_edge("planner", "de_anonymize_plan")
    # 从去匿名化流转到计划拆解
    agent_workflow.add_edge("de_anonymize_plan", "break_down_plan")
    # 从计划拆解流转到任务处理
    agent_workflow.add_edge("break_down_plan", "task_handler")
    # 从任务处理流转到检索或回答
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
