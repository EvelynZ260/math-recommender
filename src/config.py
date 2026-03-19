import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Assets
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'mysql_qa_with_vectors.pkl')  # 更名避免与旧 scienceqa 混淆
FAISS_INDEX_PATH    = os.path.join(DATA_DIR, 'mysql_qa.faiss')
EMB_MATRIX_PATH     = os.path.join(DATA_DIR, 'mysql_qa_embeddings.npy')

# --- Embedding Model (used only in offline build) ---
SBERT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# --- MySQL ---
MYSQL_USER = os.getenv("DB_USER", "root")
MYSQL_PASS = os.getenv("DB_PASS", "123456")
MYSQL_HOST = os.getenv("DB_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("DB_PORT", "3306"))
MYSQL_DB   = "collaborative_practice_platform"
MYSQL_VIEW = "qa_view"

# --- Recommendation Params ---
TOP_K_RECOMMENDATIONS     = 5
RECENT_WRONG_ANSWERS_COUNT= 10
FAISS_SEARCH_CANDIDATES   = 200

# 追加到现有 config.py 末尾
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
AGENT_MEMORY_WINDOW = 10   # 保留最近对话轮数

# --- Java endpoint to forward the results ---
JAVA_ENDPOINT = "http://localhost:8080/api/getRecommend"