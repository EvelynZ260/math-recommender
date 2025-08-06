import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import torch

from src import config

# Use tqdm with pandas
tqdm.pandas()


def build_question_assets():
    """
    Phase 0: Offline asset generation for the ScienceQA dataset.
    Downloads data from Hugging Face, generates embeddings, and saves the assets.
    """
    print("--- Phase 0: Starting Offline Asset Generation for ScienceQA ---")

    # 1. Load Data from Hugging Face Hub
    print(f"Loading '{config.HF_DATASET_NAME}' dataset from Hugging Face...")
    # We will use the 'train' split which is the largest.
    # The 'default' config will be used.
    dataset = load_dataset(config.HF_DATASET_NAME, 'default', split='train')
    df = dataset.to_pandas()

    # Add a unique QuestionID for easy reference
    df['QuestionID'] = df.index
    print(f"Loaded {len(df)} questions from the 'train' split.")

    # 2. Feature Engineering & Cleaning
    # The 'question' column is already clean text, so no HTML cleaning needed.
    # We will combine the question and the multiple-choice options for a richer embedding.
    df['choices_text'] = df['choices'].apply(lambda x: ' '.join(x))
    df['full_content'] = df['question'] + ' ' + df['choices_text']

    # 3. Generate Semantic Vectors using Sentence-BERT
    print(f"Loading Sentence-BERT model: '{config.SBERT_MODEL_NAME}'...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    sbert_model = SentenceTransformer(config.SBERT_MODEL_NAME, device=device)

    print("Generating semantic embeddings for all questions...")
    semantic_vectors = sbert_model.encode(
        df['full_content'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )

    # 4. Encode Structural Features
    print("Encoding structural features (grade, topic)...")
    df['grade'] = df['grade'].fillna('Unknown').astype(str)
    df['topic'] = df['topic'].fillna('Unknown').astype(str)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    structural_features = encoder.fit_transform(df[['grade', 'topic']])

    # 5. Combine All Features into a Single Vector
    print("Combining semantic and structural features into final vectors...")
    # NOTE: The 'difficulty' feature is not available in this dataset.
    # The vector is now: v_i = [BERT(content) + OneHot(grade/topic)]
    combined_vectors = np.hstack((semantic_vectors, structural_features))

    # Add the final vector to the DataFrame
    df['vector'] = list(combined_vectors)

    # 6. Save Processed Data
    print(f"Saving processed data with vectors to {config.PROCESSED_DATA_PATH}...")
    # Select only the columns needed for the recommender
    final_df = df[['QuestionID', 'question', 'grade', 'topic', 'vector']]
    final_df.to_pickle(config.PROCESSED_DATA_PATH)

    print("--- Offline Asset Generation Complete! ---")
    print(f"Processed {len(df)} questions.")
    print(f"Vector dimension: {combined_vectors.shape[1]}")


if __name__ == '__main__':
    build_question_assets()