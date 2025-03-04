from langchain_ollama import OllamaChatModel

model = OllamaChatModel(model="qwen2.5:14b")

response = model.invoke("你好")
print(response)
