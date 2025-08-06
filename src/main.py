import os
import pandas as pd
from src import config
from src.preprocess import build_question_assets
from src.recommendation_engine import Recommender


def simulate_user_activity() -> pd.DataFrame:
    """Creates a sample DataFrame of a user's answer history for ScienceQA."""
    print("\n--- Simulating User Activity ---")
    # Using QuestionIDs from the new dataset (indices from 0 upwards)
    user_answers_data = {
        'UserID': [1, 1, 1, 1, 1, 1],
        'QuestionID': [100, 250, 500, 1005, 1240, 2010],
        'IsCorrect': [True, False, True, False, False, True]
    }
    df = pd.DataFrame(user_answers_data)
    print("Sample user history created:")
    print(df)
    return df


def main():
    """Main function to run the recommendation process with ScienceQA."""
    # --- PHASE 0: PREPARATION ---
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        print("Processed data not found. Running the offline asset generation.")
        build_question_assets()
    else:
        print("Found pre-processed question assets. Skipping generation.")

    # --- PHASE 1: RECOMMENDATION ---
    recommender = Recommender()
    user_history = simulate_user_activity()
    target_user_id = 1
    target_user_history = user_history[user_history['UserID'] == target_user_id]

    print(f"\n--- Generating recommendations for User {target_user_id} ---")
    recommendations = recommender.recommend(
        user_history_df=target_user_history,
        k=config.TOP_K_RECOMMENDATIONS
    )

    # --- DISPLAY RESULTS ---
    print(f"\nTop {config.TOP_K_RECOMMENDATIONS} Recommendations for User {target_user_id}:\n")

    if recommendations.empty:
        print("No recommendations to display.")
        return

    # Show the questions the user got wrong for context
    wrong_questions = recommender.questions_df[
        recommender.questions_df['QuestionID'].isin(
            target_user_history[target_user_history['IsCorrect'] == False]['QuestionID']
        )
    ]
    print("Based on the following incorrectly answered questions:")
    for _, row in wrong_questions.iterrows():
        print(f"  - [ID: {row['QuestionID']}] [Grade: {row['grade']}] [Topic: {row['topic']}]")
        print(f"    {row['question'][:100]}...")
    print("-" * 60)

    # Print the recommended questions
    print("Recommended questions are:")
    for _, row in recommendations.iterrows():
        print(f"  Recommendation (Score: {row['score']:.4f}):")
        print(f"    - Question ID: {row['QuestionID']}")
        print(f"    - Content: {row['question'][:100]}...")
        print(f"    - Grade: {row['grade']}")
        print(f"    - Topic: {row['topic']}\n")


if __name__ == '__main__':
    main()