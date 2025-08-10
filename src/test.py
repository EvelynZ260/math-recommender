import os
import logging
import pandas as pd
from src import config
from src.recommendation_engine import Recommender

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def simulate_user_activity() -> pd.DataFrame:
    """
    构造一个用户历史（含对错）；你也可以换成真实用户记录。
    列名兼容 QuestionID + IsCorrect。
    """
    data = {
        'UserID':     [1]*12,
        'QuestionID': [101,102,103,104,105,106,107,108,109,110, 111,112],
        'IsCorrect':  [1,0,1,0,0,1,0,0,1,0,  0,1]  # 混合，最近的在最后
    }
    return pd.DataFrame(data)

def main():
    # 初始化推荐器（会加载 data/scienceqa_with_vectors.pkl）
    rec = Recommender()

    # 模拟用户
    user_hist = simulate_user_activity()
    target_user = 1
    uh = user_hist[user_hist['UserID'] == target_user][['QuestionID','IsCorrect']]

    # 最近错题 ID（取最近 N 道）
    wrong = uh[uh['IsCorrect'].isin([0, False])].tail(config.RECENT_WRONG_ANSWERS_COUNT)
    wrong_ids = wrong['QuestionID'].tolist()

    print("\n=== Wrong question stems ===")
    missing = []
    for qid in wrong_ids:
        if qid in rec.questions_df.index:
            row = rec.questions_df.loc[qid]
            subj = row.get('subject', '')
            topic = row.get('topic', '')
            grade = row.get('grade', '')
            stem = str(row.get('question', ''))
            print(f"\n[ID {qid}] [Subject: {subj}] [Topic: {topic}] [Grade: {grade}]")
            print(stem[:200] + ("..." if len(stem) > 200 else ""))
        else:
            missing.append(qid)
    if missing:
        print("\n[WARN] Not found in questions_df:", missing)

    # 生成推荐
    print("\n=== Generating Recommendations ===")
    recs = rec.recommend(user_history_df=uh, k=config.TOP_K_RECOMMENDATIONS)

    if recs.empty:
        print("\nNo recommendations.")
        return

    print("\n[Recommendations]:")
    for _, row in recs.iterrows():
        print(f"- ID: {row['QuestionID']} | Score: {row['score']:.4f} | Topic: {row['topic']} | Grade: {row.get('grade')}")
        print(f"  Q: {str(row['question'])[:100]}...")

if __name__ == "__main__":
    main()
