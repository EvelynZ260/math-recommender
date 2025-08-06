import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src import config


class Recommender:
    """
    A recommendation engine that suggests questions based on a user's
    performance history, adapted for the ScienceQA dataset.
    """

    def __init__(self):
        print("Initializing Recommender...")
        try:
            self.questions_df = pd.read_pickle(config.PROCESSED_DATA_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Processed data not found at {config.PROCESSED_DATA_PATH}. "
                f"Please run `src/preprocess.py` first to generate the assets."
            )

        self.question_vectors = np.vstack(self.questions_df['vector'].values)
        print("Recommender initialized successfully.")

    def _get_user_weakness_vector(self, wrong_question_ids: list) -> np.ndarray:
        """
        Calculates the user's "weakness vector" by averaging the vectors
        of the questions they answered incorrectly.
        """
        if not wrong_question_ids:
            return np.zeros(self.question_vectors.shape[1])

        wrong_vectors = self.question_vectors[self.questions_df['QuestionID'].isin(wrong_question_ids)]

        if wrong_vectors.shape[0] == 0:
            return np.zeros(self.question_vectors.shape[1])

        return np.mean(wrong_vectors, axis=0)

    def recommend(
            self,
            user_history_df: pd.DataFrame,
            k: int = config.TOP_K_RECOMMENDATIONS
    ) -> pd.DataFrame:
        """
        Generates top-k question recommendations for a user.
        The scoring is based purely on cosine similarity.
        """
        # 1. Identify user's incorrectly answered questions
        wrong_answers = user_history_df[user_history_df['IsCorrect'] == False]
        wrong_question_ids = wrong_answers['QuestionID'].tolist()

        print(f"\nUser has {len(wrong_question_ids)} wrong answers. Generating weakness vector...")

        # 2. Generate the user's current weakness vector
        user_vector = self._get_user_weakness_vector(wrong_question_ids)

        # 3. Calculate final recommendation score using cosine similarity
        # The score is purely based on semantic and structural similarity, as there is no difficulty score.
        final_scores = cosine_similarity(user_vector.reshape(1, -1), self.question_vectors)[0]

        recommendations = self.questions_df.copy()
        recommendations['score'] = final_scores

        # 4. Filter out questions the user has already answered
        answered_question_ids = user_history_df['QuestionID'].tolist()
        recommendations = recommendations[~recommendations['QuestionID'].isin(answered_question_ids)]

        # 5. Sort and return top-K results
        recommendations = recommendations.sort_values(by='score', ascending=False)

        return recommendations.head(k)