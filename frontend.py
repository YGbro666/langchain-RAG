import streamlit as st
import requests
import os

# streamlit run frontend.py --server.port=8082
# 设置页面标题
st.set_page_config(page_title="LLM Chat Interface", layout="wide")

# 初始化session state用于存储对话历史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

BACKEND_URL = "http://localhost:8084"
LANGCHAIN_URL = "http://localhost:8083"
Ollama_URL = "http://localhost:11434"

# 自定义CSS样式
st.markdown(
    """
<style>
.user-message {
    background-color: #e6f3ff;
    padding: 15px;
    border-radius: 10px;
    margin: 5px 0;
}
.ai-message {
    background-color: #f0f0f0;
    padding: 15px;
    border-radius: 10px;
    margin: 5px 0;
}
.chat-container {
    margin-bottom: 20px;
}
</style>
""",
    unsafe_allow_html=True,
)


def load_model(model_name):
    """加载模型"""
    try:
        # Ollama API 要求的负载格式
        payload = {"name": model_name, "stream": False}

        # 使用 Ollama 的模型加载 API
        response = requests.post(
            f"{Ollama_URL}/api/pull",  # Ollama 的拉取模型 API 端点
            json=payload,
            timeout=300,  # 5分钟超时
        )

        if response.status_code == 200:
            return {"status": "success", "message": f"模型 {model_name} 加载成功"}
        else:
            return {"status": "error", "message": f"模型加载失败: {response.text}"}

    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "message": "无法连接到Ollama服务，请确保Ollama服务正在运行",
        }
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "message": "加载模型超时，这可能是因为模型较大或网络问题，请稍后重试",
        }
    except Exception as e:
        return {"status": "error", "message": f"发生错误: {str(e)}"}


def chat_with_model(prompt, max_tokens=512, temperature=1.0):
    """与模型对话（使用 agent 模式）"""
    try:
        response = requests.post(
            f"{LANGCHAIN_URL}/agent",  # 改为使用 agent 端点
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=300,  # 添加 5 分钟超时
        )
        return response.json()
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "message": "请求超时，这可能是因为问题较复杂或需要多个工具处理，请稍后重试",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def check_model_status():
    """检查模型状态"""
    try:
        response = requests.get(f"{BACKEND_URL}/model_status")
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 页面标题
st.title("🤖 LLM Chat Interface")

# 侧边栏 - 模型选择和加载
with st.sidebar:
    st.header("模型设置")

    # 修改为Ollama支持的模型选项
    model_options = {
        "Qwen-14B": "qwen2.5:14b",
        "Qwen-7B": "qwen2.5:7b",
        "Llama2": "llama2",
        "Mistral": "mistral",
        "Neural-Chat": "neural-chat",
        "CodeLlama": "codellama",
    }

    selected_model = st.selectbox("选择模型", options=list(model_options.keys()))

    # 加载模型按钮
    if st.button("加载模型"):
        with st.spinner("正在加载模型，这可能需要几分钟时间，请耐心等待..."):
            result = load_model(model_options[selected_model])
            if isinstance(result, dict):
                if result["status"] == "success":
                    st.success(result["message"])
                else:
                    st.error(result.get("message", "未知错误"))
            else:
                st.error("无效的响应格式")

    # 显示模型状态
    st.subheader("模型状态")
    status = check_model_status()
    st.write(f"模型已加载: {'✅' if status.get('model_loaded') else '❌'}")

# 主界面 - 对话区域
st.header("对话区域")

# 显示对话历史
st.markdown("### 对话历史")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(
            f'<div class="user-message">👤 用户: {message["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="ai-message">🤖 AI: {message["content"]}</div>',
            unsafe_allow_html=True,
        )

# 清除历史记录按钮
if st.button("清除历史记录"):
    st.session_state.chat_history = []
    st.rerun()

# 参数设置
col1, col2 = st.columns(2)
with col1:
    max_tokens = st.slider("最大生成长度", 64, 2048, 512)
with col2:
    temperature = st.slider("温度 (随机性)", 0.1, 2.0, 1.0)

# 用户输入
user_input = st.text_area("请输入您的问题:", height=100)

# Agent对话按钮
if st.button("Agent对话"):
    if not user_input.strip():
        st.warning("请输入问题！")
    else:
        # 添加用户消息到历史记录
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("正在思考并处理您的问题..."):
            response = chat_with_model(
                user_input, max_tokens=max_tokens, temperature=temperature
            )
            if response.get("status") == "success":
                # 添加AI回复到历史记录
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response["response"]}
                )
                # 重新加载页面以显示新消息
                st.rerun()
            else:
                st.error(f"错误: {response.get('message', '未知错误')}")

# 更新页面底部信息
st.markdown("---")
st.markdown("""
💡 提示：
- 先在左侧选择并加载模型
- Agent对话：智能代理，可以使用多种工具
- 可以调整生成长度和随机性参数
""")
