from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from modelscope import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict
import uvicorn
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import  requests
import os  
# python model-loader.py

app = FastAPI(title="LLM API Service")

# 添加 CORS 中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型和tokenizer
global_model = None
global_tokenizer = None

model_id = "iic/nlp_gte_sentence-embedding_chinese-base"
pipeline_se = pipeline(
    Tasks.sentence_embedding, model=model_id, sequence_length=512
)  # sequence_length 代表最大文本长度，默认值为128


# 请求体模型
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 1.0
    system_prompt: Optional[str] = "You are a helpful assistant."


# 响应体模型
class ChatResponse(BaseModel):
    status: str
    response: str


class EmbeddingRequest(BaseModel):
    source_sentence: List[str]
    sentences_to_compare: List[str]


class EmbeddingResponse(BaseModel):
    text_embedding: List[List[float]]
    scores: List[float]


@app.get("/")
async def root():
    """健康检查接口"""
    return {"status": "service is running", "model_loaded": global_model is not None}



@app.post("/load_model")
def load_model(request: dict):
    """加载Ollama模型"""
    try:
        model_name = request.get("model_name")

        # 尝试拉取模型
        response = requests.post(
            "http://localhost:8085/api/pull", json={"name": model_name}, stream=True
        )

        # 检查响应
        if response.status_code == 200:
            # 设置当前使用的模型名称
            os.environ["CURRENT_MODEL"] = model_name
            return {"status": "success", "message": f"模型 {model_name} 加载成功"}
        else:
            return {"status": "error", "message": f"模型加载失败: {response.text}"}

    except Exception as e:
        return {"status": "error", "message": f"加载模型时发生错误: {str(e)}"}

# 点击发送
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    与模型进行对话

    Args:
        request: 包含提示词和其他参数的请求体
    """
    try:
        if global_model is None or global_tokenizer is None:
            raise HTTPException(
                status_code=400, detail="Model not loaded. Please load a model first."
            )

        # 构建对话消息
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.prompt},
        ]

        # 使用tokenizer处理输入
        text = global_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 准备模型输入
        model_inputs = global_tokenizer([text], return_tensors="pt").to(
            global_model.device
        )

        # 生成回答
        generated_ids = global_model.generate(
            **model_inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
        )

        # 提取生成的文本
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 解码生成的文本
        response = global_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return ChatResponse(status="success", response=response)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )


@app.get("/model_status")
async def get_model_status():
    """获取当前模型状态"""
    return {
        "model_loaded": global_model is not None,
        "tokenizer_loaded": global_tokenizer is not None,
    }


@app.post("/embedding", response_model=EmbeddingResponse)
async def embedding(request: EmbeddingRequest):
    """
    获取文本向量
    """
    try:
        input_data = {
            "source_sentence": request.source_sentence,
            "sentences_to_compare": request.sentences_to_compare,
        }
        result = pipeline_se(input_data)
        return EmbeddingResponse(
            text_embedding=result["text_embedding"], scores=result["scores"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"处理请求时发生错误: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "modelscope_model:app",
        host="0.0.0.0",
        port=8091,
    )
