import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import faiss
from src import config

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ---------------------------- #
# 1) Load questions from MySQL #
# ---------------------------- #
def load_questions_from_mysql() -> pd.DataFrame:
    url = (f"mysql+pymysql://{config.MYSQL_USER}:{config.MYSQL_PASS}"
           f"@{config.MYSQL_HOST}:{config.MYSQL_PORT}/{config.MYSQL_DB}?charset=utf8mb4")
    engine = create_engine(url, pool_pre_ping=True)
    df = pd.read_sql(text(f"SELECT * FROM {config.MYSQL_VIEW}"), engine)

    # 期望有 qa_id 作为主键
    if "qa_id" not in df.columns:
        raise KeyError("Expected column 'qa_id' in MySQL view")

    df = df.set_index("qa_id", drop=True).sort_index()

    # 重命名/对齐部分字段名（可按你的列名再扩展）
    df.rename(columns={
        "answer_index": "AnswerIndex",
        "question": "question",
        "grade": "grade",
        "topic": "topic",
        "subject": "subject",
        "category": "category",
        "choices": "choices_json"
    }, inplace=True)

    return df

# -------------------------------- #
# 2) Normalize/parse choices field #
# -------------------------------- #
def parse_choices_to_text_and_list(raw) -> Tuple[str, list]:
    """
    输入：JSON string / dict / list
    输出：
      choices_text: "A. ...\nB. ...\n..."
      choices_list: ["...", "...", ...]
    """
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            return "", []
    if isinstance(data, dict):
        order = ["A", "B", "C", "D", "E", "F"]
        pairs = [(k, data[k]) for k in order if k in data]
        text = "\n".join([f"{k}. {v}" for k, v in pairs])
        lst  = [v for _, v in pairs]
        return text, lst
    if isinstance(data, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        text = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(data)])
        return text, data
    return "", []

# ----------------------- #
# 3) Build text for embed #
# ----------------------- #
def build_text_for_embedding(row: pd.Series) -> str:
    parts = [
        f"[Subject] {row.get('subject', '')}" if pd.notna(row.get('subject')) else "",
        f"[Topic] {row.get('topic', '')}" if pd.notna(row.get('topic')) else "",
        f"[Category] {row.get('category', '')}" if pd.notna(row.get('category')) else "",
        str(row.get("question", "")).strip(),
        str(row.get("choices_text", "")).strip(),
    ]
    return "\n".join([p for p in parts if p])

# ------------------------- #
# 4) Embedding + FAISS save #
# ------------------------- #
def build_question_assets():
    logging.info("Loading questions from MySQL...")
    df = load_questions_from_mysql()
    logging.info(f"Loaded {len(df)} rows from MySQL.")

    # 显式的 QuestionID（与 MySQL 主键一致）
    df["QuestionID"] = df.index.astype(int)

    # 解析选项
    logging.info("Parsing choices...")
    choices_texts, choices_lists = [], []
    for raw in df["choices_json"]:
        t, l = parse_choices_to_text_and_list(raw)
        choices_texts.append(t)
        choices_lists.append(l)
    df["choices_text"] = choices_texts
    df["choices_list"] = choices_lists

    # 构造用于嵌入的文本
    logging.info("Building text_for_embedding...")
    df["text_for_embedding"] = df.apply(build_text_for_embedding, axis=1)

    # 生成向量
    logging.info("Encoding with SentenceTransformer...")
    model = SentenceTransformer(config.SBERT_MODEL_NAME)
    texts = df["text_for_embedding"].fillna("").tolist()
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    emb = emb.astype('float32')

    # 构建 FAISS（内积 = 余弦相似）
    logging.info("Building FAISS index...")
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    assert index.ntotal == emb.shape[0]

    # 保存（打印绝对路径+计数）
    pkl_path   = os.path.abspath(config.PROCESSED_DATA_PATH)
    faiss_path = os.path.abspath(config.FAISS_INDEX_PATH)
    emb_path   = os.path.abspath(config.EMB_MATRIX_PATH)

    logging.info(f"[Assets][WRITE] output ->\n  PKL  : {pkl_path}\n  FAISS: {faiss_path}\n  EMB  : {emb_path}")
    logging.info(f"[Assets][WRITE] rows={len(df)}, min_id={df['QuestionID'].min()}, max_id={df['QuestionID'].max()}")

    df.to_pickle(pkl_path)
    faiss.write_index(index, faiss_path)
    np.save(emb_path, emb)

    logging.info("Assets saved successfully.")

if __name__ == "__main__":
    build_question_assets()