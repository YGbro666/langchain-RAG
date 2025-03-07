import requests
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Optional, Iterator
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import (
    Tool,
    initialize_agent,
)
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_community.document_loaders.helpers import detect_file_encodings
import logging
import sys

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)

embedding_dimension = 768

app = FastAPI()

BACKEND_URL = "http://localhost:8084"
LANGCHAIN_URL = "http://localhost:8083"
Ollama_URL = "http://localhost:11434"


def get_embeddings_from_local_api(
    source_sentence: List[str], sentences_to_compare: List[str]
) -> List[List[float]]:
    url = BACKEND_URL + "/embedding"
    payload = EmbeddingRequest(
        source_sentence=source_sentence, sentences_to_compare=sentences_to_compare
    )
    response = requests.post(url, json=payload.dict())
    response_data = response.json()
    embeddings = EmbeddingResponse(**response_data)
    return embeddings.text_embedding


# 替换为使用本地 API 获取嵌入向量
class LocalEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 调用本地 API 获取文档嵌入
        return get_embeddings_from_local_api(
            source_sentence=texts, sentences_to_compare=[]
        )

    def embed_query(self, query: str) -> List[float]:
        # 调用本地 API 获取查询嵌入
        return get_embeddings_from_local_api(
            source_sentence=[query], sentences_to_compare=[]
        )[0]


# 重写TextLoader
class CustomTextLoader(TextLoader):
    def __init__(
        self,
        file_path: str,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
    ):
        super().__init__(file_path, encoding, autodetect_encoding)
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )

    def lazy_load(self) -> Iterator[Document]:
        """加载文件并分割文本"""
        text = ""
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                text = f.read()
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    logger.debug(f"Trying encoding: {encoding.encoding}")
                    try:
                        with open(self.file_path, encoding=encoding.encoding) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        # 使用文本分割器处理文本
        metadata = {"source": str(self.file_path)}
        doc = Document(page_content=text, metadata=metadata)
        for chunk in self.text_splitter.split_documents([doc]):
            yield chunk


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 1.0
    system_prompt: Optional[str] = "You are a helpful assistant."


class ChatResponse(BaseModel):
    status: str
    response: str


class ChatHistory(BaseModel):
    timestamp: str
    prompt: str
    response: str
    status: str


class ChatHistoryManager:
    def __init__(self, history_file: str = "chat_history.json"):
        # 使用绝对路径
        self.history_file = os.path.abspath(history_file)
        logger.info(f"历史记录文件路径: {self.history_file}")
        self._ensure_history_file()

    def _ensure_history_file(self):
        try:
            if not os.path.exists(self.history_file):
                logger.info(f"创建新的历史记录文件: {self.history_file}")
                with open(self.history_file, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
            else:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    json.load(f)
                logger.info(f"成功加载现有历史记录文件: {self.history_file}")
        except Exception as e:
            logger.error(f"初始化历史记录文件时发生错误: {str(e)}")
            # 如果出错，尝试重新创建文件
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def add_chat(self, prompt: str, response: str, status: str):
        try:
            logger.info(f"正在添加新的聊天记录到文件: {self.history_file}")
            # 读取现有历史记录
            history = []
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)

            # 添加新的对话记录
            chat_entry = ChatHistory(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                prompt=prompt,
                response=response,
                status=status,
            ).dict()

            history.append(chat_entry)

            # 写入更新后的历史记录
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            logger.info("成功保存聊天记录")

        except Exception as e:
            logger.error(f"保存聊天历史时发生错误: {str(e)}")
            raise

    def get_history(self, limit: Optional[int] = None) -> List[ChatHistory]:
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            if limit:
                history = history[-limit:]
            return [ChatHistory(**entry) for entry in history]
        except Exception as e:
            logger.error(f"读取聊天历史时发生错误: {str(e)}")
            return []


# 创建聊天历史管理器实例
chat_history_manager = ChatHistoryManager()


class EmbeddingRequest(BaseModel):
    source_sentence: List[str]
    sentences_to_compare: List[str]


class EmbeddingResponse(BaseModel):
    text_embedding: List[List[float]]
    scores: List[float]


@app.post("/RAGchat", response_model=ChatResponse)
def RAGchat(request: ChatRequest):
    try:
        # 加载向量存储
        vectorstore = FAISS.load_local(
            "my_index/output/hongloumeng_vectorstore",
            LocalEmbedding(),
            allow_dangerous_deserialization=True,
        )

        # 创建检索器，设置搜索参数
        retriever = vectorstore.as_retriever()

        # 创建LLM实例
        llm = OllamaLLM(
            model="qwen2.5:14b",
            base_url=Ollama_URL,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # 构建问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,  # 返回源文档信息
        )

        # 执行查询
        result = qa_chain({"query": request.prompt})

        # 提取答案和源文档信息
        answer = result["result"]
        source_docs = result.get("source_documents", [])

        # 构建响应
        response = f"{answer}\n\n参考文档：\n"
        for i, doc in enumerate(source_docs, 1):
            response += f"{i}. {doc.metadata.get('source', '未知来源')}\n"

        # 保存聊天历史
        chat_history_manager.add_chat(
            prompt=request.prompt, response=response, status="success"
        )

        return ChatResponse(status="success", response=response)

    except Exception as e:
        error_response = f"处理请求时发生错误: {str(e)}"
        # 保存错误记录到聊天历史
        chat_history_manager.add_chat(
            prompt=request.prompt, response=error_response, status="error"
        )
        return ChatResponse(status="error", response=error_response)


@app.post("/RAG_write_index", response_model=ChatResponse)
def RAG_write_index():
    try:
        vectorstore = []
        for root, sub_folders, files in os.walk("my_index/input/hongloumeng"):
            for file in files:
                loader = CustomTextLoader(os.path.join(root, file))
                docs = loader.load()
                embeddings = LocalEmbedding()
                if vectorstore == []:
                    vectorstore = FAISS.from_documents(docs, embeddings)
                else:
                    vectorstore.add_documents(docs)
        vectorstore.save_local("my_index/output/hongloumeng_vectorstore")
        return ChatResponse(status="success", response="索引写入成功")
    except Exception as e:
        return ChatResponse(status="error", response=f"处理请求时发生错误: {str(e)}")


# class Agentic_RAG_Tool(BaseTool):
#     name:str = "Agentic_RAG_Tool"
#     description:str = "必须使用一次此工具。输入应该是具体的问题或查询关键词。"
#     return_direct:bool = True
#     def _run(self, query: str) -> str:
#         request = ChatRequest(prompt=query)
#         chat_response = RAGchat(request)
#         return chat_response.response


def RAG_write_index_wrapper(*args, **kwargs) -> str:
    """包装RAG_write_index函数,忽略所有输入参数"""
    response = RAG_write_index()
    return response.response


def RAGchat_tool(query: str) -> str:
    request = ChatRequest(prompt=query)
    chat_response = RAGchat(request)
    return chat_response.response


class RAGchat_tool_input(BaseModel):
    question: str = Field(description="需要回答的问题或查询关键词")


def format_chat_history(history: List[ChatHistory], max_history: int = 5) -> str:
    """格式化最近的聊天历史记录"""
    if not history:
        return ""

    # 只取最近的几条记录
    recent_history = history[-max_history:]
    formatted_history = "以下是最近的对话历史：\n\n"

    for entry in recent_history:
        formatted_history += f"用户: {entry.prompt}\n"
        formatted_history += f"助手: {entry.response}\n\n"

    return formatted_history


@app.post("/agent", response_model=ChatResponse)
def agent(request: ChatRequest):
    try:
        # 获取历史记录
        history = chat_history_manager.get_history(limit=5)  # 获取最近5条记录
        chat_history = format_chat_history(history)

        # 构建带有历史记录的提示
        prompt_with_history = f"""请基于以下历史对话记录和当前问题进行回答。注意保持对话的连贯性和上下文的相关性。

{chat_history}
当前问题: {request.prompt}
"""

        llm = OllamaLLM(model="qwen2.5:14b", base_url=Ollama_URL)
        Agentic_RAG_Tool = Tool.from_function(
            name="Agentic_RAG_Tool",
            description="用于查询知识库的工具。注意：必须先使用Index_Creater_Tool创建索引后才能使用此工具！如果索引不存在将会报错。输入应该是具体的问题或查询关键词。",
            func=RAGchat_tool,
            return_direct=True,
            args_schema=RAGchat_tool_input,
        )
        Index_Creater_Tool = Tool.from_function(
            name="Index_Creater_Tool",
            description="创建知识库索引的工具。在进行任何对于知识库的查询之前，必须首先调用此工具来创建索引。这是第一步！该工具不需要任何输入,只需要调用即可。",
            func=RAG_write_index_wrapper,
            return_direct=False,
        )
        tools = [Agentic_RAG_Tool, Index_Creater_Tool]
        agent = initialize_agent(
            llm=llm, tools=tools, handle_parsing_errors=True, verbose=True
        )
        result = agent.run(prompt_with_history)  # 使用带有历史记录的提示

        # 保存聊天历史
        chat_history_manager.add_chat(
            prompt=request.prompt, response=result, status="success"
        )

        return ChatResponse(status="success", response=result)
    except Exception as e:
        error_response = f"处理请求时发生错误: {str(e)}"
        # 保存错误记录到聊天历史
        chat_history_manager.add_chat(
            prompt=request.prompt, response=error_response, status="error"
        )
        return ChatResponse(status="error", response=error_response)


# 添加新的历史记录查询端点
@app.get("/chat_history")
def get_chat_history(limit: Optional[int] = None):
    try:
        history = chat_history_manager.get_history(limit)
        return {"status": "success", "history": [entry.dict() for entry in history]}
    except Exception as e:
        return {"status": "error", "message": f"获取聊天历史时发生错误: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(
        "langchain_demo:app",
        host="localhost",
        port=8083,
        reload=True,  # 开发模式下启用热重载
    )
