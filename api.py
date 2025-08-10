import os
import pandas as pd
from flask import Flask, request, jsonify
import logging
import requests

from src.recommendation_engine import Recommender
from src.preprocess import build_question_assets
from src import config

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
logging.info("Flask App initialized.")

recommender = None

def init_recommender():
    global recommender
    # 若资产不存在，先构建
    need_build = (not os.path.exists(config.PROCESSED_DATA_PATH)
                  or not os.path.exists(config.FAISS_INDEX_PATH)
                  or not os.path.exists(config.EMB_MATRIX_PATH))
    if need_build:
        logging.warning("Assets not found. Building offline assets...")
        build_question_assets()

    recommender = Recommender()
    logging.info("Recommender loaded successfully.")

# --- 初始化 ---
try:
    init_recommender()
except Exception as e:
    logging.error(f"Failed to initialize recommender: {e}", exc_info=True)
    recommender = None

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    logging.info("Received a request on /recommend.")
    if recommender is None:
        return jsonify({"error": "Recommender service is not available."}), 500

    # 1) 解析输入（顶层 records）
    try:
        payload = request.get_json(force=True, silent=False)
        logging.info(f"Raw JSON received: {payload}")
        records = payload.get("records", [])
        if not isinstance(records, list):
            return jsonify({"error": "Invalid format: 'records' must be a list."}), 400
        if not all(isinstance(x, dict) for x in records):
            return jsonify({"error": "'records' must be a list of objects."}), 400
    except Exception as e:
        logging.error(f"JSON parse error: {e}", exc_info=True)
        return jsonify({"error": "Invalid JSON"}), 400

    # 2) 转 DataFrame + 规范列
    try:
        df = pd.DataFrame(records)
        # 兼容 questionId/isCorrect
        if not {'questionId', 'isCorrect'}.issubset(df.columns):
            return jsonify({"error": "Missing keys: 'questionId' and 'isCorrect'"}), 400
        df = df.rename(columns={'questionId': 'QuestionID', 'isCorrect': 'IsCorrect'})
        df['QuestionID'] = df['QuestionID'].astype(int)
        df['IsCorrect'] = df['IsCorrect'].apply(lambda x: bool(int(x)) if isinstance(x, (int, str)) else bool(x))
        logging.info(f"user_history_df preview:\n{df.head()}")
    except Exception as e:
        logging.error(f"DataFrame conversion error: {e}", exc_info=True)
        return jsonify({"error": "Invalid data in 'records'"}), 400

    # 3) 生成推荐
    try:
        rec_df = recommender.recommend(user_history_df=df)
        recommended_ids = [int(q) for q in rec_df['QuestionID'].tolist()]
        logging.info(f"Generated {len(recommended_ids)} recommendations: {recommended_ids}")
    except Exception as e:
        logging.error(f"Recommendation error: {e}", exc_info=True)
        return jsonify({"error": "Internal error while generating recommendations"}), 500

    # 4) 转发给 Java（只发 questionIds）
    java_ok = False
    try:
        resp = requests.post(
            config.JAVA_ENDPOINT,
            json={"questionIds": recommended_ids},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        logging.info(f"Forwarded to Java. Status={resp.status_code}, Body={resp.text[:300]}")
        java_ok = 200 <= resp.status_code < 300
    except Exception as e:
        logging.warning(f"Failed to POST to Java: {e}", exc_info=True)

    # 5) 返回给调用方
    return jsonify({"questionIds": recommended_ids}), 200

if __name__ == '__main__':
    # 0.0.0.0 便于其他容器/主机访问；端口 5001 避开 Java 的 8080
    app.run(host='0.0.0.0', port=5001, debug=True)