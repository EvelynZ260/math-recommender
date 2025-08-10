import numpy as np
import pandas as pd
from collections import Counter
import faiss
from src import config

class Recommender:
    """
    使用最近错题(默认10)的topic分布做权重分配，生成5条个性化多样化推荐。
    """
    def __init__(self):
        # 载入题目资产（DataFrame 的 index 为 QuestionID/qa_id）
        self.questions_df = pd.read_pickle(config.PROCESSED_DATA_PATH)
        self.faiss_index  = faiss.read_index(config.FAISS_INDEX_PATH)

        # 重建全部向量（为快速取出若干题的向量）
        all_vecs = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
        # 注意：reconstruct_n 依赖索引类型；IndexFlatIP 可以工作
        self.question_vectors_map = {self.questions_df.index[i]: all_vecs[i] for i in range(self.faiss_index.ntotal)}

        # 为了将 FAISS 内部顺序 ↔ DataFrame 行对应起来
        self._rowid_to_qid = dict(enumerate(self.questions_df.index.tolist()))

    # ---------- 内部工具 ---------- #
    def _get_user_weakness_vector(self, seed_ids: list) -> np.ndarray:
        if not seed_ids:
            return np.zeros(self.faiss_index.d, dtype='float32')
        vecs = [self.question_vectors_map[qid] for qid in seed_ids if qid in self.question_vectors_map]
        if not vecs:
            return np.zeros(self.faiss_index.d, dtype='float32')
        user_vec = np.mean(vecs, axis=0).astype('float32')
        return user_vec

    def _faiss_search(self, user_vec: np.ndarray, top_m: int):
        v = user_vec.reshape(1, -1).astype('float32')
        faiss.normalize_L2(v)  # 保险：保持归一化
        D, I = self.faiss_index.search(v, top_m)
        return D[0], I[0]

    def _allocate_quota(self, topic_weights: dict, total_k: int) -> dict:
        """
        按权重分配名额，向下取整，最少1，之后用小数部分做校正，保证总数=total_k。
        """
        topics = list(topic_weights.keys())
        raw = {t: topic_weights[t] * total_k for t in topics}
        quota = {t: int(np.floor(raw[t])) for t in topics}
        for t in topics:
            if quota[t] == 0:
                quota[t] = 1
        diff = total_k - sum(quota.values())
        if diff > 0:
            frac = sorted([(t, raw[t] - np.floor(raw[t])) for t in topics], key=lambda x: x[1], reverse=True)
            for i in range(diff):
                quota[frac[i % len(frac)][0]] += 1
        elif diff < 0:
            frac = sorted([(t, raw[t] - np.floor(raw[t])) for t in topics], key=lambda x: x[1])
            need = -diff
            i = 0
            while need > 0 and any(quota[t] > 1 for t,_ in frac):
                t = frac[i % len(frac)][0]
                if quota[t] > 1:
                    quota[t] -= 1
                    need -= 1
                i += 1
        return quota

    # ---------- Topic 定向召回 ---------- #
    def _recommend_for_topic(self, topic: str, recent_wrong_ids: set,
                             already_picked: set, need_k: int):
        if need_k <= 0:
            return []

        # 若该 topic 没有错题，则用全部错题做用户向量；否则只用该 topic 的错题
        topic_wrong = [qid for qid in recent_wrong_ids
                       if str(self.questions_df.loc[qid]['topic']) == str(topic)]
        seed_ids = topic_wrong if topic_wrong else list(recent_wrong_ids)

        user_vec = self._get_user_weakness_vector(seed_ids)
        D, I = self._faiss_search(user_vec, config.FAISS_SEARCH_CANDIDATES)

        recs = []
        for row_i, dist in zip(I, D):
            qid = self._rowid_to_qid[row_i]
            if qid in already_picked or qid in recent_wrong_ids:
                continue
            row = self.questions_df.loc[qid]
            if str(row['topic']) != str(topic):
                continue
            recs.append({
                'QuestionID': qid,
                'score': float(dist),
                'topic': row['topic'],
                'grade': row.get('grade', None),
                'question': row['question']
            })
            if len(recs) >= need_k:
                break
        return recs

    # ---------- 主推荐 ---------- #
    def recommend(self, user_history_df: pd.DataFrame, k: int = config.TOP_K_RECOMMENDATIONS) -> pd.DataFrame:
        """
        使用最近10道错题：计算topic权重→按权重分配名额→分topic召回→不足回填→TOP-k。
        兼容列名：QuestionID/isCorrect 或 questionId/isCorrect。
        """
        df = user_history_df.copy()
        # 兼容大小写
        cols_map = {c.lower(): c for c in df.columns}
        q_col = cols_map.get('questionid') or cols_map.get('question_id')
        ic_col = cols_map.get('iscorrect')

        if q_col is None or ic_col is None:
            raise ValueError("user_history_df 需要列：QuestionID 与 IsCorrect/isCorrect")

        # 只取错题 & 取最近 N（按输入顺序的最后 N 条）
        wrong_df = df[df[ic_col].isin([0, False])]
        recent_wrong = wrong_df.tail(config.RECENT_WRONG_ANSWERS_COUNT)
        recent_wrong_ids = [int(x) for x in recent_wrong[q_col].tolist()]
        recent_wrong_ids = [qid for qid in recent_wrong_ids if qid in self.questions_df.index]
        recent_wrong_ids_set = set(recent_wrong_ids)

        # 没有错题：用“全局中心向量”简单召回
        if not recent_wrong_ids:
            center = np.mean([self.question_vectors_map[q] for q in self.questions_df.index[:1000]], axis=0).astype('float32')
            D, I = self._faiss_search(center, k)
            res = []
            for row_i, d in zip(I, D):
                qid = self._rowid_to_qid[row_i]
                r = self.questions_df.loc[qid]
                res.append({'QuestionID': qid, 'score': float(d), 'topic': r['topic'], 'grade': r.get('grade', None), 'question': r['question']})
            return pd.DataFrame(res[:k])

        # 统计 topic 权重
        topics = [self.questions_df.loc[qid]['topic'] for qid in recent_wrong_ids]
        cnt = Counter([t for t in topics if pd.notna(t)])
        total = sum(cnt.values())
        topic_weights = {t: cnt[t] / total for t in cnt}

        # 分配配额
        quota = self._allocate_quota(topic_weights, k)

        picked = set()
        results = []

        # 分topic召回
        for t, need in quota.items():
            recs = self._recommend_for_topic(t, recent_wrong_ids_set, picked, need)
            for r in recs:
                picked.add(r['QuestionID'])
            results.extend(recs)

        # 回填：不限制topic，只避开已选与错题
        if len(results) < k:
            user_vec = self._get_user_weakness_vector(recent_wrong_ids)
            D, I = self._faiss_search(user_vec, config.FAISS_SEARCH_CANDIDATES)
            for row_i, dist in zip(I, D):
                qid = self._rowid_to_qid[row_i]
                if qid in picked or qid in recent_wrong_ids_set:
                    continue
                r = self.questions_df.loc[qid]
                results.append({
                    'QuestionID': qid, 'score': float(dist),
                    'topic': r['topic'], 'grade': r.get('grade', None), 'question': r['question']
                })
                picked.add(qid)
                if len(results) >= k:
                    break

        results = sorted(results, key=lambda x: x['score'], reverse=True)[:k]
        return pd.DataFrame(results)
