import streamlit as st

from functions_for_pipeline_train import create_agent
import streamlit.components.v1 as components


def update_placeholders_and_graph(agent_state_value, placeholders, graph_placeholder, previous_values, previous_state):
    # 根据当前状态更新Streamlit应用中的占位符和图表。
    current_state = agent_state_value.get("curr_state")

    # 更新图表
    if current_state:
        net = create_network_graph(current_state)
        graph_html = save_and_display_graph(net)
        graph_placeholder.empty()
        with graph_placeholder.container():
            components.html(graph_html, height=400, scrolling=True)

    # 仅当状态已改变时（即我们已访问完上一个节点）才更新占位符
    if current_state != previous_state and previous_state is not None:
        for key, placeholder in placeholders.items():
            if key in previous_values and previous_values[key] is not None:
                if isinstance(previous_values[key], list):
                    formatted_value = "\n".join(
                        [f"{i+1}. {item}" for i, item in enumerate(previous_values[key])])
                else:
                    formatted_value = previous_values[key]
                placeholder.markdown(f"{formatted_value}")


def execute_plan_and_print_steps(inputs, plan_and_execute_app,
                                 placeholders, graph_placeholder, recursion_limit=25):
    # 执行计划并在Streamlit应用中打印出步骤。
    config = {"recursion_limit": recursion_limit}
    agent_state_value = None
    progress_bar = st.progress(0)
    step = 0
    previous_state = None
    previous_values = {key: None for key in placeholders}

    try:
        # 让智能体（Agent）开始处理任务，并且以‘流式输出（Stream）’的方式，把整个推理和执行过程中的每一步结果，像流水一样逐条返回给我
        for plan_output in plan_and_execute_app.stream(inputs, config=config):
            step += 1
            # 遍历智能体当前步吐出的所有数据，并且只关心它所包含的具体状态内容（即具体的值），而忽略掉节点名称
            for _, agent_state_value in plan_output.items():
                previous_values, previous_state = update_placeholders_and_graph(
                    agent_state_value, placeholders, graph_placeholder,
                    previous_values, previous_state
                )

                progress_bar.progress(step / recursion_limit)

                if step >= recursion_limit:
                    break

        # 循环结束后，用最终状态更新占位符
        for key, placeholder in placeholders.items():
            if key in previous_values and previous_values[key] is not None:
                if isinstance(previous_values[key], list):
                    formatted_value = "\n".join(
                        [f"{i+1}. {item}" for i, item in enumerate(previous_values[key])])
                else:
                    formatted_value = previous_values[key]

                placeholder.markdown(f"{formatted_value}")

        response = agent_state_value.get(
            'response', "No response found.") if agent_state_value else "No response found."
    except Exception as e:
        response = f"An error occurred: {str(e)}"
        st.error(f"Error: {e}")

    return response


def main():
    # 主方法
    st.set_page_config(layout="wide")

    st.title("实时代理执行可视化")

    # 加载现有的代理创建函数
    plan_and_execute_app = create_agent()

    # 得到用户的问题
    question = st.text_input("输入你的问题:", "那个帮助恶棍的教授在教什么课？")

    if st.button("Run Agent"):
        inputs = {"question": question}

        # 为图标创建一行
        st.markdown("**图**")
        graph_placeholder = st.empty()

        # 为其他变量创建三列
        col1, col2, col3 = st.columns([1, 1, 4])

        with col1:
            st.markdown("**计划**")
        with col2:
            st.markdown("**过去的步骤**")
        with col3:
            st.markdown("**聚合上下文**")

        # 为每一列初始化占位符
        placeholder = {
            "plan": col1.empty(),
            "past_steps": col2.empty(),
            "aggregated_context": col3.empty
        }

        response = execute_plan_and_print_steps(
            inputs, plan_and_execute_app, placeholder, graph_placeholder, recursion_limit=45)
        # response = "回答了"
        st.write("结束回答:")
        st.write(response)


if __name__ == "__main__":
    main()
