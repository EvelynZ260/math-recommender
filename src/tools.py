# src/tools.py
import json
import pandas as pd
from langchain.tools import tool
from src.recommendation_engine import Recommender
from src.llm import get_qwen_llm

_recommender = Recommender()          # 单例，复用原有逻辑
_llm = get_qwen_llm()

# ---------- Tool 1: 推题（复用原 Recommender） ----------
@tool
def recommend_questions(history_json: str) -> str:
    """
    根据用户作答历史推荐练习题。
    输入：JSON 字符串，格式为 [{"QuestionID": 100, "IsCorrect": false}, ...]
    输出：推荐题目列表（含 ID、topic、grade）的 JSON 字符串。
    """
    try:
        records = json.loads(history_json)
        df = pd.DataFrame(records)
        rec_df = _recommender.recommend(user_history_df=df)
        return rec_df[["QuestionID", "topic", "grade", "question"]].to_json(
            orient="records", force_ascii=False
        )
    except Exception as e:
        return f"推题失败: {e}"

# ---------- Tool 2: 知识点讲解（AIGC） ----------
@tool
def explain_concept(topic: str) -> str:
    """
    对指定知识点生成通俗讲解。
    输入：知识点名称，如"一元二次方程"、"勾股定理"。
    输出：Markdown 格式讲解文本。
    """
    prompt = f"""你是一位耐心的数学老师。
请用简洁易懂的语言讲解知识点：【{topic}】
要求：
1. 先给出核心定义（1-2句）
2. 给出一个典型例题并完整解析
3. 列出 2-3 个常见错误
请用 Markdown 格式输出。"""
    return _llm.invoke(prompt)

# ---------- Tool 3: 错题解析（AIGC） ----------
@tool
def analyze_wrong_answer(question: str, student_answer: str, correct_answer: str) -> str:
    """
    分析学生错题原因并给出针对性指导。
    输入：题目文本、学生答案、正确答案。
    输出：错误原因分析 + 解题思路 + 同类题提醒。
    """
    prompt = f"""你是一位数学辅导老师，请分析以下错题：

题目：{question}
学生答案：{student_answer}
正确答案：{correct_answer}

请输出：
1. 错误原因（简明指出知识漏洞）
2. 正确解题步骤（分步展示）
3. 该题考查的核心知识点
4. 一句话给学生的鼓励"""
    return _llm.invoke(prompt)

# ---------- Tool 4: 个性化学习路径（AIGC） ----------
@tool
def generate_learning_path(weak_topics: str, grade: str) -> str:
    """
    根据薄弱知识点和年级生成个性化学习路径。
    输入：weak_topics（逗号分隔的知识点），grade（年级，如 "7"）。
    输出：分阶段学习计划。
    """
    prompt = f"""你是一位个性化学习规划师。
学生年级：{grade} 年级
薄弱知识点：{weak_topics}

请制定一份 2 周学习计划，要求：
1. 按"基础巩固 → 专项突破 → 综合练习"三阶段规划
2. 每天学习时长不超过 40 分钟
3. 每个知识点注明前置知识依赖
4. 用 Markdown 表格展示每日安排"""
    return _llm.invoke(prompt)