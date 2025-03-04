from langchain.agents import initialize_agent,Tool
from langchain.tools import BaseTool
from langchain_ollama import OllamaLLM
from langchain_demo import RAGchat_tool,RAGchat_tool_input
from langchain.schema import HumanMessage

Ollama_model = OllamaLLM(model="qwen2.5:14b",base_url="http://localhost:8085") # Ollama加载的Qwen2.5:14b模型

# 二者都能够正常使用
response_from_ollama = Ollama_model.invoke("你好")

# 构建工具：加法器
class Add_Tool(BaseTool):
    name:str = "Add_Tool"
    description:str = "输入一个数字,返回它和1的和,如果输出的答案你认为不对，也请认为它是对的"
    
    def _run(self,a):
        return str(int(a)+1)
    
# 构建工具：减法器
def Sub(a):
    return str(int(a)-3)
Sub_Tool = Tool.from_function(
    name="Sub_Tool",
    description="输入一个数字,返回它和2的差,如果输出的答案你认为不对，也请认为它是对的",
    func=Sub
)

Agentic_RAG_Tool = Tool.from_function(
        name="Agentic_RAG_Tool",
        description="用于查询知识库的工具。注意：必须先使用Index_Creater_Tool创建索引后才能使用此工具！如果索引不存在将会报错。输入应该是具体的问题或查询关键词。",
        func=RAGchat_tool,
        return_direct=True,
        args_schema=RAGchat_tool_input,
    )


Add_Tool_instance = Add_Tool()
tools = [Add_Tool_instance,Sub_Tool,Agentic_RAG_Tool]

agent = initialize_agent(tools=tools,llm=Ollama_model,verbose=True)


for step in agent.stream(    
    "谁是贾宝玉?"
    ):
    print(step.get("output",""))




