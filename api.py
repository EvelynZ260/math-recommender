import os
import pandas as pd
from flask import Flask, request, jsonify
import logging

from src.recommendation_engine import Recommender
from src.preprocess import build_question_assets
from src import config

# --- Initialization ---
# Configure logging for debugging purposes
logging.basicConfig(level=logging.INFO)

# 1. Initialize Flask App
app = Flask(__name__)
logging.info("Flask App initialized.")

# --- Global model loading ---
# This is a crucial performance optimization step.
# We load the model and data into memory at service startup instead of loading them for every request.
# This ensures fast API response times.
recommender = None

try:
    # 2. Check if processed data exists, if not, generate it
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        logging.warning(f"Processed data file not found at {config.PROCESSED_DATA_PATH}.")
        logging.info("Starting offline asset generation process. This may take a while...")
        build_question_assets()
        logging.info("Asset generation complete.")

    # 3. Instantiate the recommender, loading data and model
    recommender = Recommender()
    logging.info("Recommender loaded successfully.")
except Exception as e:
    logging.error(f"Failed to initialize recommender: {e}", exc_info=True)
    # If initialization fails, recommender will be None and API requests will return an error.


# --- API Route Definitions ---
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    Main recommendation API endpoint.
    Receives the user's question-answer history and returns a list of recommended question IDs.
    """
    global recommender
    logging.info("Received a request on /recommend.")

    # Check if recommender is available
    if recommender is None:
        logging.error("Recommender is not available. Check server logs for initialization errors.")
        return jsonify({"error": "Recommender service is not available."}), 500

    # 1. Get POST JSON data
    try:
        data = request.get_json()
        if not data or 'history' not in data:
            logging.warning("Request is missing 'history' field.")
            return jsonify({"error": "Invalid request format. Missing 'history' field."}), 400
    except Exception as e:
        logging.error(f"Failed to parse JSON request: {e}")
        return jsonify({"error": "Invalid JSON format."}), 400

    user_history_list = data['history']
    logging.info(f"Received user history with {len(user_history_list)} records.")

    # 2. Convert JSON data to Pandas DataFrame (required by model)
    if not isinstance(user_history_list, list) or not all(isinstance(item, dict) for item in user_history_list):
        return jsonify({"error": "'history' must be a list of objects."}), 400

    # 'isCorrect': 0 = incorrect (False), 1 = correct (True)
    try:
        user_history_df = pd.DataFrame(user_history_list)
        # Rename fields to match model's expected column names
        user_history_df.rename(columns={'questionId': 'QuestionID', 'isCorrect': 'IsCorrect'}, inplace=True)
        user_history_df['IsCorrect'] = user_history_df['IsCorrect'].astype(bool)
    except (KeyError, TypeError) as e:
        logging.error(f"Error converting history to DataFrame: {e}")
        return jsonify({"error": "Invalid data structure in 'history'. Expected keys: 'questionId', 'isCorrect'."}), 400

    # 3. Generate recommendations
    try:
        recommendations_df = recommender.recommend(user_history_df=user_history_df)

        # 4. Extract IDs as Python ints for JSON serialization
        recommended_ids = [int(id) for id in recommendations_df['QuestionID'].tolist()]
        logging.info(f"Generated {len(recommended_ids)} recommendations.")

        # 5. Return JSON response
        return jsonify({"questionIds": recommended_ids})
    except Exception as e:
        logging.error(f"An error occurred during recommendation generation: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while generating recommendations."}), 500


# --- Run server ---
if __name__ == '__main__':
    # Use a different port from Java backend (e.g., 5001)
    # host='0.0.0.0' allows external access (e.g., from another Docker container or VM)
    app.run(host='0.0.0.0', port=5001, debug=True)
