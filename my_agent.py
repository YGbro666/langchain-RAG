from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("damo/ModelScope-Agent-14B", revision = 'v1.0.0',trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("damo/ModelScope-Agent-14B", revision = 'v1.0.0',device_map="auto", trust_remote_code=True,fp16=True).eval()
model.generation_config = GenerationConfig.from_pretrained("damo/ModelScope-Agent-14B",revision = 'v1.0.0', trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参


system = """
 <|system|>:当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按
照json格式给出，必须包含api_name、url、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。然后你需要根据插
件API调用结果生成合理的答复；若无需调用插件，则直接给出对应回复即可：

1. {"name": "modelscope_image-generation", "description": "图片生成服务，针对文本输入，生成对应的图片，插图等", "parameters": [{"name": "text", "description": "用户输入的文本信息", "required": true}]}

2. {"name": "modelscope_speech-generation", "description": "朗读文本内容，将文本转语音服务，将文字转换为自然而逼真的语音，可配置男声/女声", "parameters": [{"name": "input", "description": "要转成语音的文本", "required": true}, {"name": "gender", "description": "用户身份", "required": true}]}
"""

response, history = model.chat(tokenizer, "生成一张插图，关于狐狸和猎人的。", history=None, system=system)
print(response)
response, history = model.chat(tokenizer, "假如你是李清照，现在和我对话，你最喜欢的词是哪一首，语音回复我。", history=history, system=system)
print(response)
