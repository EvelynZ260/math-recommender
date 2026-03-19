# src/llm.py
import os
from langchain_community.llms import Tongyi          # pip install langchain-community dashscope
from langchain_core.language_models import BaseLLM

def get_qwen_llm(streaming: bool = False) -> BaseLLM:
    """
    使用阿里云 DashScope API 调用 Qwen-7B。
    本地部署版本见下方注释。
    """
    return Tongyi(
        model_name="qwen-7b-chat",                   # 或 qwen-plus / qwen-turbo
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        streaming=streaming,
        temperature=0.7,
        max_tokens=1024,
    )

# ---------- 本地部署版本（用 transformers） ----------
# from langchain_community.llms import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#
# def get_qwen_llm():
#     model_id = "Qwen/Qwen-7B-Chat"
#     tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
#     return HuggingFacePipeline(pipeline=pipe)