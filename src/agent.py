# src/agent.py
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from src.llm import get_qwen_llm
from src.tools import (
    recommend_questions,
    explain_concept,
    analyze_wrong_answer,
    generate_learning_path,
)

TOOLS = [recommend_questions, explain_concept, analyze_wrong_answer, generate_learning_path]

SYSTEM_PROMPT = """你是一个个性化学习推荐 AI，能帮助学生找到合适的练习题、讲解知识点、分析错题、规划学习路径。

你可以使用以下工具：
{tools}

工具名称列表：{tool_names}

对话历史：
{chat_history}

用户输入：{input}

请按以下格式思考和行动：
Thought: 分析用户意图，决定是否调用工具
Action: 工具名称（必须是工具列表之一）
Action Input: 工具的输入参数
Observation: 工具返回结果
...（可多轮）
Thought: 已有足够信息
Final Answer: 给用户的完整回复

{agent_scratchpad}"""

def build_agent(session_id: str) -> AgentExecutor:
    """为每个 session 构建独立 Agent（含独立 Memory）。"""
    llm = get_qwen_llm()
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=10,                        # 保留最近 10 轮
        return_messages=False,
    )
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,  # Qwen 偶尔格式不规范，容错
        max_iterations=6,
    )

# 全局 session 池（生产环境换成 Redis）
_session_pool: dict[str, AgentExecutor] = {}

def get_or_create_agent(session_id: str) -> AgentExecutor:
    if session_id not in _session_pool:
        _session_pool[session_id] = build_agent(session_id)
    return _session_pool[session_id]