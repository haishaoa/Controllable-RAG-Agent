import streamlit as st

from functions_for_pipeline import create_agent


def main():
    # 主方法
    st.set_page_config(layout="wide")

    st.title("实时代理执行可视化")

    # 加载现有的代理创建函数
    plan_and_execute_app = create_agent()

    # 得到用户的问题
    question = st.text_input("输入你的问题:","那个帮助恶棍的教授在教什么课？")

    if st.button("Run Agent"):
        inputs = {"question":question}

        # 为图标创建一行

if __name__ == "__main__":
    main()
