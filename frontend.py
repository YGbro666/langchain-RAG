import streamlit as st
import requests
import urllib.parse
import os

# streamlit run frontend.py --server.port=8082
# 设置页面标题
st.set_page_config(page_title="LLM Chat Interface", layout="wide")

# 在文件开头添加 session_state 的初始化
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = "default"

BACKEND_URL = "http://localhost:8084"
LANGCHAIN_URL = "http://localhost:8083"
Ollama_URL = "http://localhost:11434"


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


def create_rag_index():
    """创建 RAG 知识库索引"""
    try:
        response = requests.post(
            f"{LANGCHAIN_URL}/RAG_write_index",
            timeout=300,  # 5分钟超时
        )
        return response.json()
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "message": "创建索引超时，这可能是因为文件较多或硬件资源受限，请稍后重试",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def rag_chat(prompt, max_tokens=512, temperature=1.0, conversation_id="default"):
    """使用 RAG 方式与模型对话"""
    try:
        response = requests.post(
            f"{LANGCHAIN_URL}/RAGchat",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "conversation_id": conversation_id,  # 添加会话ID
            },
            timeout=300,
        )
        return response.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "请求超时，请稍后重试"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def upload_file(uploaded_file):
    """处理文件上传"""
    try:
        # 确保目录存在
        save_path = "my_index/input/extra"
        os.makedirs(save_path, exist_ok=True)

        # 保存文件
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return {"status": "success", "message": f"文件 {uploaded_file.name} 上传成功"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def clear_chat_history(conversation_id):
    """清除特定会话的历史记录"""
    try:
        response = requests.post(f"{LANGCHAIN_URL}/clear_memory/{conversation_id}")
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

    # 添加一个分隔线
    st.markdown("---")

    # 添加知识库索引部分
    st.subheader("知识库管理")
    if st.button("构建知识库索引"):
        with st.spinner("正在构建知识库索引，这可能需要几分钟时间，请耐心等待..."):
            result = create_rag_index()
            if result["status"] == "success":
                st.success("知识库索引构建成功！")
            else:
                st.error(f"知识库索引构建失败: {result.get('message', '未知错误')}")

    # 在侧边栏中添加文件上传部分
    st.markdown("---")
    st.subheader("文件上传")

    # 文件上传组件
    uploaded_files = st.file_uploader(
        "上传文本文件到知识库",
        accept_multiple_files=True,
        type=["txt", "md", "doc", "docx"],  # 限制文件类型
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"正在上传文件 {uploaded_file.name}..."):
                result = upload_file(uploaded_file)
                if result["status"] == "success":
                    st.success(result["message"])
                else:
                    st.error(f"上传失败: {result['message']}")

        # 提示用户重建索引
        st.info("文件上传完成后，请点击'构建知识库索引'按钮更新知识库")

    # 在侧边栏中添加会话管理部分
    st.markdown("---")
    st.subheader("会话管理")

    # 会话ID输入
    new_conversation_id = st.text_input(
        "会话ID",
        value=st.session_state.conversation_id,
        help="输入一个唯一的会话标识符",
    )

    # 更新会话ID
    if new_conversation_id != st.session_state.conversation_id:
        st.session_state.conversation_id = new_conversation_id
        st.success(f"已切换到会话: {new_conversation_id}")

    # 清除会话历史按钮
    if st.button("清除当前会话历史"):
        with st.spinner("正在清除会话历史..."):
            result = clear_chat_history(st.session_state.conversation_id)
            if result["status"] == "success":
                st.success(result["message"])
            else:
                st.error(f"清除失败: {result.get('message', '未知错误')}")

# 主界面 - 对话区域
st.header("对话区域")

# 参数设置
col1, col2 = st.columns(2)
with col1:
    max_tokens = st.slider("最大生成长度", 64, 2048, 512)
with col2:
    temperature = st.slider("温度 (随机性)", 0.1, 2.0, 1.0)

# 用户输入
user_input = st.text_area("请输入您的问题:", height=100)

# 创建两个按钮的列
col1, col2 = st.columns(2)

# Agent对话按钮
with col1:
    if st.button("Agent对话"):
        if not user_input.strip():
            st.warning("请输入问题！")
        else:
            with st.spinner("正在思考并处理您的问题..."):
                response = chat_with_model(
                    user_input, max_tokens=max_tokens, temperature=temperature
                )
                if response.get("status") == "success":
                    st.markdown("### Agent回答:")
                    st.write(response["response"])
                else:
                    st.error(f"错误: {response.get('message', '未知错误')}")

# RAG对话按钮
with col2:
    if st.button("RAG对话"):
        if not user_input.strip():
            st.warning("请输入问题！")
        else:
            with st.spinner("正在从知识库中检索相关信息..."):
                response = rag_chat(
                    user_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    conversation_id=st.session_state.conversation_id,  # 添加会话ID
                )
                if response.get("status") == "success":
                    st.markdown("### RAG回答:")
                    st.write(response["response"])
                else:
                    st.error(f"错误: {response.get('message', '未知错误')}")

# 更新页面底部信息
st.markdown("---")
st.markdown("""
💡 提示：
- 先在左侧选择并加载模型
- 使用"构建知识库索引"按钮更新知识库
- 可以在左侧设置会话ID，管理不同的对话上下文
- 选择合适的对话方式：
  - Agent对话：智能代理，可以使用多种工具
  - RAG对话：直接从知识库中检索答案，支持上下文记忆
- 可以调整生成长度和随机性参数
- 使用"清除当前会话历史"按钮重置对话上下文
""")
