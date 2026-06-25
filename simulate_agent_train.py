import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from functions_for_pipeline_train import create_agent


@st.cache_resource(show_spinner="正在加载 Agent...")
def load_agent():
    """缓存 Agent，避免 Streamlit 每次刷新页面都重新构建向量库和 LangGraph。"""
    return create_agent()


def create_network_graph(current_state):
    """创建智能体当前状态的网络图可视化。"""

    # Streamlit 里不是 Jupyter Notebook，notebook 应该设为 False。
    # cdn_resources="in_line" 可以避免 pyvis 的本地 CDN 警告。
    net = Network(
        directed=True,
        notebook=False,
        cdn_resources="in_line",
        height="250px",
        width="100%",
    )

    # 禁用物理模拟，避免节点乱动
    net.toggle_physics(False)

    nodes = [
        {"id": "anonymize_question", "label": "anonymize_question", "x": 0, "y": 0},
        {
            "id": "planner",
            "label": "planner",
            "x": 175 * 1.75,
            "y": -100,
        },
        {
            "id": "de_anonymize_plan",
            "label": "de_anonymize_plan",
            "x": 350 * 1.75,
            "y": -100,
        },
        {
            "id": "break_down_plan",
            "label": "break_down_plan",
            "x": 525 * 1.75,
            "y": -100,
        },
        {
            "id": "task_handler",
            "label": "task_handler",
            "x": 700 * 1.75,
            "y": 0,
        },
        {
            "id": "retrieve_chunks",
            "label": "retrieve_chunks",
            "x": 875 * 1.75,
            "y": 200,
        },
        {
            "id": "retrieve_summaries",
            "label": "retrieve_summaries",
            "x": 875 * 1.75,
            "y": 100,
        },
        {
            "id": "retrieve_book_quotes",
            "label": "retrieve_book_quotes",
            "x": 875 * 1.75,
            "y": 0,
        },
        {
            "id": "answer",
            "label": "answer",
            "x": 875 * 1.75,
            "y": -100,
        },
        {
            "id": "replan",
            "label": "replan",
            "x": 1050 * 1.75,
            "y": 0,
        },
        {
            "id": "can_be_answered_already",
            "label": "can_be_answered_already",
            "x": 1225 * 1.75,
            "y": 0,
        },
        {
            "id": "get_final_answer",
            "label": "get_final_answer",
            "x": 1400 * 1.75,
            "y": 0,
        },
    ]

    edges = [
        ("anonymize_question", "planner"),
        ("planner", "de_anonymize_plan"),
        ("de_anonymize_plan", "break_down_plan"),
        ("break_down_plan", "task_handler"),
        ("task_handler", "retrieve_chunks"),
        ("task_handler", "retrieve_summaries"),
        ("task_handler", "retrieve_book_quotes"),
        ("task_handler", "answer"),
        ("retrieve_chunks", "replan"),
        ("retrieve_summaries", "replan"),
        ("retrieve_book_quotes", "replan"),
        ("answer", "replan"),
        ("replan", "can_be_answered_already"),
        ("replan", "break_down_plan"),
        ("can_be_answered_already", "get_final_answer"),
    ]

    # 添加节点
    for node in nodes:
        color = "#00FF00" if node["id"] == current_state else "#FF69B4"
        net.add_node(
            node["id"],
            label=node["label"],
            x=node["x"],
            y=node["y"],
            color=color,
            physics=False,
            font={"size": 22},
        )

    # 添加边
    for source, target in edges:
        net.add_edge(source, target, color="#808080")

    # 设置边样式
    net.options.edges.smooth.type = "straight"
    net.options.edges.width = 1.5

    return net


def save_and_display_graph(net):
    """将网络图保存为 HTML 字符串，供 Streamlit 展示。"""

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            tmp_path = tmp_file.name

        net.write_html(tmp_path, notebook=False, open_browser=False)

        with open(tmp_path, "r", encoding="utf-8") as f:
            return f.read()

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def format_value(value):
    """把 list / dict / 普通字符串格式化成适合 markdown 展示的内容。"""

    if isinstance(value, list):
        return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(value))

    if isinstance(value, dict):
        return "```json\n" + str(value) + "\n```"

    return str(value)


def update_placeholders_and_graph(
    agent_state_value,
    placeholders,
    graph_placeholder,
    previous_values,
    previous_state,
):
    """根据当前状态更新 Streamlit 应用中的占位符和图表。"""

    if not isinstance(agent_state_value, dict):
        return previous_values, previous_state

    current_state = agent_state_value.get("curr_state")

    # 更新图表
    if current_state:
        net = create_network_graph(current_state)
        graph_html = save_and_display_graph(net)

        graph_placeholder.empty()

        with graph_placeholder.container():
            components.html(graph_html, height=400, scrolling=True)

    # 状态切换时，把上一轮保存的值显示出来
    if current_state != previous_state and previous_state is not None:
        for key, placeholder in placeholders.items():
            value = previous_values.get(key)

            if value is not None:
                placeholder.markdown(format_value(value))

    # 保存当前 state 的展示字段，下一轮状态切换时再刷新
    for key in placeholders:
        if key in agent_state_value:
            previous_values[key] = agent_state_value[key]

    return previous_values, current_state or previous_state


def execute_plan_and_print_steps(
    inputs,
    plan_and_execute_app,
    placeholders,
    graph_placeholder,
    recursion_limit=25,
):
    """执行 Agent，并把 LangGraph 的流式步骤展示到 Streamlit 页面。"""

    config = {"recursion_limit": recursion_limit}

    agent_state_value = None
    progress_bar = st.progress(0)
    step = 0
    previous_state = None
    previous_values = {key: None for key in placeholders}

    try:
        for plan_output in plan_and_execute_app.stream(inputs, config=config):
            step += 1

            for _, state_value in plan_output.items():
                agent_state_value = state_value

                previous_values, previous_state = update_placeholders_and_graph(
                    agent_state_value,
                    placeholders,
                    graph_placeholder,
                    previous_values,
                    previous_state,
                )

            progress_bar.progress(min(step / recursion_limit, 1.0))

            if step >= recursion_limit:
                st.warning(f"已达到 recursion_limit={recursion_limit}，流程被停止。")
                break

        # 循环结束后，把最后一次缓存的值也显示出来
        for key, placeholder in placeholders.items():
            value = previous_values.get(key)

            if value is not None:
                placeholder.markdown(format_value(value))

        if not agent_state_value:
            return "No response found."

        response = agent_state_value.get("response", "No response found.")

        if isinstance(response, dict):
            return response.get("answer", str(response))

        return response

    except Exception as e:
        st.error(f"Error: {e}")
        return f"An error occurred: {e}"


def main():
    """Streamlit 主入口。"""

    st.set_page_config(layout="wide")

    st.title("实时代理执行可视化")

    question = st.text_input("输入你的问题:", "那个帮助恶棍的教授在教什么课？")

    if st.button("Run Agent"):
        plan_and_execute_app = load_agent()

        inputs = {"question": question}

        st.markdown("**图**")
        graph_placeholder = st.empty()

        col1, col2, col3 = st.columns([1, 1, 4])

        with col1:
            st.markdown("**计划**")

        with col2:
            st.markdown("**过去的步骤**")

        with col3:
            st.markdown("**聚合上下文**")

        placeholders = {
            "plan": col1.empty(),
            "past_steps": col2.empty(),
            "aggregated_context": col3.empty(),
        }

        response = execute_plan_and_print_steps(
            inputs,
            plan_and_execute_app,
            placeholders,
            graph_placeholder,
            recursion_limit=45,
        )

        st.write("结束回答:")
        st.write(response)


if __name__ == "__main__":
    main()