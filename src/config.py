import os

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to store the processed DataFrame with question vectors for the new dataset
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'scienceqa_with_vectors.pkl')


# --- Model & Dataset Configuration ---
# Hugging Face dataset name
HF_DATASET_NAME = 'derek-thomas/ScienceQA'

# Sentence-BERT model for generating embeddings
SBERT_MODEL_NAME = 'shibing624/text2vec-base-chinese' # This is a Chinese model, let's switch to a multilingual or English one
# SBERT_MODEL_NAME = 'all-MiniLM-L6-v2' # A good general-purpose English model
SBERT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


# --- Recommendation Parameters ---
# Number of recommendations to generate
TOP_K_RECOMMENDATIONS = 5