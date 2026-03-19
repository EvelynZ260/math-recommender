import re
from collections import Counter
import numpy as np
import pandas as pd
import faiss
from src import config
from typing import Optional

class Recommender:
    """
    使用最近错题(默认10)的 topic 分布做权重分配，生成 K 条个性化多样化推荐；
    且强约束：推荐题目的 grade 不超过最近错题的最大年级（grade_cap）。
    """

    def __init__(self):
        # 载入题目资产（DataFrame 的 index 为 QuestionID/qa_id）
        self.questions_df = pd.read_pickle(config.PROCESSED_DATA_PATH)
        self.faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)

        # 重建全部向量（为快速取出若干题的向量）
        # 注意：reconstruct_n 依赖索引类型；IndexFlatIP 可以工作
        all_vecs = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
        self.question_vectors_map = {
            self.questions_df.index[i]: all_vecs[i]
            for i in range(self.faiss_index.ntotal)
        }

        # 将 FAISS 内部顺序 ↔ DataFrame 行对应起来
        self._rowid_to_qid = dict(enumerate(self.questions_df.index.tolist()))

    # ---------- 工具 ---------- #
    def _grade_to_num(self, g):
        """
        将 grade 解析为数字：
        7, '7', 'grade7', 'Grade 7' -> 7
        解析失败返回 None
        """
        if g is None or (isinstance(g, float) and np.isnan(g)):
            return None
        if isinstance(g, (int, np.integer)):
            return int(g)
        s = str(g).strip().lower()
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None

    def _get_user_weakness_vector(self, seed_ids: list) -> np.ndarray:
        if not seed_ids:
            return np.zeros(self.faiss_index.d, dtype="float32")
        vecs = [
            self.question_vectors_map[qid]
            for qid in seed_ids
            if qid in self.question_vectors_map
        ]
        if not vecs:
            return np.zeros(self.faiss_index.d, dtype="float32")
        user_vec = np.mean(vecs, axis=0).astype("float32")
        return user_vec

    def _faiss_search(self, user_vec: np.ndarray, top_m: int):
        v = user_vec.reshape(1, -1).astype("float32")
        faiss.normalize_L2(v)  # 保险：保持归一化（内积≈余弦）
        D, I = self.faiss_index.search(v, top_m)
        return D[0], I[0]

    def _allocate_quota(self, topic_weights: dict, total_k: int) -> dict:
        """
        按权重分配名额：
        - 若出现的 topic 数量 > K：仅保留权重最高的 K 个，每个 1 条（总和==K）
        - 否则：沿用原策略（floor + 至少1 + 小数部分校正，确保总和==K）
        """
        topics = list(topic_weights.keys())
        if not topics or total_k <= 0:
            return {}

        # 情况 A：topic 数量 > K，只取权重最高的 K 个，每个 1 条
        if len(topics) > total_k:
            topics_sorted = sorted(
                topics, key=lambda t: (-topic_weights[t], str(t))
            )
            chosen = topics_sorted[:total_k]
            return {t: 1 for t in chosen}  # 总和==K

        # 情况 B：topic 数量 <= K，沿用原策略
        raw = {t: topic_weights[t] * total_k for t in topics}
        quota = {t: int(np.floor(raw[t])) for t in topics}

        # 至少 1
        for t in topics:
            if quota[t] == 0:
                quota[t] = 1

        diff = total_k - sum(quota.values())
        if diff > 0:
            # 用小数部分从大到小补齐
            frac = sorted(
                [(t, raw[t] - np.floor(raw[t])) for t in topics],
                key=lambda x: x[1],
                reverse=True,
            )
            for i in range(diff):
                quota[frac[i % len(frac)][0]] += 1
        elif diff < 0:
            # 从小数部分小的开始回收，但仅从 quota>1 的里回收
            frac = sorted(
                [(t, raw[t] - np.floor(raw[t])) for t in topics],
                key=lambda x: x[1],
            )
            need = -diff
            i = 0
            while need > 0 and any(quota[t] > 1 for t, _ in frac):
                t = frac[i % len(frac)][0]
                if quota[t] > 1:
                    quota[t] -= 1
                    need -= 1
                i += 1

            # 兜底：若仍未回收到位（极少见），按权重从小到大继续减到 0
            if need > 0:
                by_weight = sorted(topics, key=lambda t: (topic_weights[t], str(t)))
                for t in by_weight:
                    if quota[t] > 0:
                        quota[t] -= 1
                        need -= 1
                        if need == 0:
                            break

        return quota

    # ---------- Topic 定向召回（含年级上限过滤） ---------- #
    def _recommend_for_topic(
        self,
        topic: str,
        recent_wrong_ids: set,
        already_picked: set,
        need_k: int,
        grade_cap: Optional[int],
    ):
        if need_k <= 0:
            return []

        # 若该 topic 没有错题，则用全部错题做用户向量；否则只用该 topic 的错题
        topic_wrong = [
            qid
            for qid in recent_wrong_ids
            if str(self.questions_df.loc[qid]["topic"]) == str(topic)
        ]
        seed_ids = topic_wrong if topic_wrong else list(recent_wrong_ids)

        user_vec = self._get_user_weakness_vector(seed_ids)
        D, I = self._faiss_search(user_vec, config.FAISS_SEARCH_CANDIDATES)

        recs = []
        for row_i, dist in zip(I, D):
            qid = self._rowid_to_qid[row_i]
            if qid in already_picked or qid in recent_wrong_ids:
                continue
            row = self.questions_df.loc[qid]

            # 主题过滤
            if str(row["topic"]) != str(topic):
                continue

            # 年级上限过滤：无法解析或超出上限则跳过
            row_grade_num = self._grade_to_num(row.get("grade", None))
            if grade_cap is not None:
                if row_grade_num is None or row_grade_num > grade_cap:
                    continue

            recs.append(
                {
                    "QuestionID": qid,
                    "score": float(dist),
                    "topic": row["topic"],
                    "grade": row.get("grade", None),
                    "question": row["question"],
                }
            )
            if len(recs) >= need_k:
                break
        return recs

    # ---------- 主推荐 ---------- #
    def recommend(
        self, user_history_df: pd.DataFrame, k: int = config.TOP_K_RECOMMENDATIONS
    ) -> pd.DataFrame:
        """
        使用最近10道错题：计算topic权重 → 按权重分配名额 → 分topic召回(含年级上限) → 不足回填(含年级上限) → TOP-k。
        兼容列名：QuestionID/isCorrect 或 question_id/isCorrect。
        """
        df = user_history_df.copy()

        # 兼容大小写
        cols_map = {c.lower(): c for c in df.columns}
        q_col = cols_map.get("questionid") or cols_map.get("question_id")
        ic_col = cols_map.get("iscorrect")

        if q_col is None or ic_col is None:
            raise ValueError("user_history_df 需要列：QuestionID 与 IsCorrect/isCorrect")

        # 只取错题 & 取最近 N（按输入顺序的最后 N 条）
        wrong_df = df[df[ic_col].isin([0, False])]
        recent_wrong = wrong_df.tail(config.RECENT_WRONG_ANSWERS_COUNT)
        recent_wrong_ids = [int(x) for x in recent_wrong[q_col].tolist()]
        recent_wrong_ids = [
            qid for qid in recent_wrong_ids if qid in self.questions_df.index
        ]
        recent_wrong_ids_set = set(recent_wrong_ids)

        # 计算最近错题中的“最大年级”作为上限
        wrong_grades = [
            self._grade_to_num(self.questions_df.loc[qid].get("grade", None))
            for qid in recent_wrong_ids
        ]
        valid_wrong_grades = [g for g in wrong_grades if g is not None]
        grade_cap = max(valid_wrong_grades) if valid_wrong_grades else None

        # 没有错题：用“全局中心向量”简单召回（不施加年级上限）
        if not recent_wrong_ids:
            center = (
                np.mean(
                    [self.question_vectors_map[q] for q in self.questions_df.index[:1000]],
                    axis=0,
                ).astype("float32")
            )
            D, I = self._faiss_search(center, k)
            res = []
            for row_i, d in zip(I, D):
                qid = self._rowid_to_qid[row_i]
                r = self.questions_df.loc[qid]
                res.append(
                    {
                        "QuestionID": qid,
                        "score": float(d),
                        "topic": r["topic"],
                        "grade": r.get("grade", None),
                        "question": r["question"],
                    }
                )
            return pd.DataFrame(res[:k])

        # 统计 topic 权重
        topics = [self.questions_df.loc[qid]["topic"] for qid in recent_wrong_ids]
        cnt = Counter([t for t in topics if pd.notna(t)])
        total = sum(cnt.values())
        topic_weights = {t: cnt[t] / total for t in cnt} if total > 0 else {}

        # 分配配额
        quota = self._allocate_quota(topic_weights, k) if topic_weights else {}

        picked = set()
        results = []

        # 分 topic 召回（带年级上限）
        for t, need in (quota or {}).items():
            recs = self._recommend_for_topic(
                t, recent_wrong_ids_set, picked, need, grade_cap
            )
            for r in recs:
                picked.add(r["QuestionID"])
            results.extend(recs)

        # 回填：不限制 topic，但仍需满足 grade <= grade_cap；只避开已选与错题
        if len(results) < k:
            user_vec = self._get_user_weakness_vector(recent_wrong_ids)
            top_m = config.FAISS_SEARCH_CANDIDATES
            D, I = self._faiss_search(user_vec, top_m)
            for row_i, dist in zip(I, D):
                qid = self._rowid_to_qid[row_i]
                if qid in picked or qid in recent_wrong_ids_set:
                    continue
                r = self.questions_df.loc[qid]

                row_grade_num = self._grade_to_num(r.get("grade", None))
                if grade_cap is not None:
                    if row_grade_num is None or row_grade_num > grade_cap:
                        continue

                results.append(
                    {
                        "QuestionID": qid,
                        "score": float(dist),
                        "topic": r["topic"],
                        "grade": r.get("grade", None),
                        "question": r["question"],
                    }
                )
                picked.add(qid)
                if len(results) >= k:
                    break

        # 如果仍不够，可以适度放大候选池再补（可注释掉）
        if len(results) < k:
            user_vec = self._get_user_weakness_vector(recent_wrong_ids)
            big_top_m = min(self.faiss_index.ntotal, config.FAISS_SEARCH_CANDIDATES * 5)
            D2, I2 = self._faiss_search(user_vec, big_top_m)
            for row_i, dist in zip(I2, D2):
                qid = self._rowid_to_qid[row_i]
                if qid in picked or qid in recent_wrong_ids_set:
                    continue
                r = self.questions_df.loc[qid]
                row_grade_num = self._grade_to_num(r.get("grade", None))
                if grade_cap is not None:
                    if row_grade_num is None or row_grade_num > grade_cap:
                        continue
                results.append(
                    {
                        "QuestionID": qid,
                        "score": float(dist),
                        "topic": r["topic"],
                        "grade": r.get("grade", None),
                        "question": r["question"],
                    }
                )
                picked.add(qid)
                if len(results) >= k:
                    break

        results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        return pd.DataFrame(results)