from statistics import mean
from src.retrieval.retrieval_pipeline import load_config
import os

def normalize(values):
    """
    Min-Max normalize.
    """
    if not values:
        return []

    v_min = min(values)
    v_max = max(values)

    if abs(v_max - v_min) < 1e-8:
        return [1.0 for _ in values]

    return [
        (v - v_min) / (v_max - v_min)
        for v in values
    ]
import numpy as np


# def compute_temporal_consistency(
#     candidate_keyframes,
#     candidate_scores,
#     candidate_ranks,
#     gap_weight=0.6,
#     score_weight=0.2,
#     rank_weight=0.2,
# ):
#     """
#     Đánh giá độ mượt của một chuỗi temporal.

#     Parameters
#     ----------
#     candidate_keyframes : list[int]

#     candidate_scores : list[float]

#     candidate_ranks : list[int]

#     Returns
#     -------
#     float
#         consistency score trong [0,1]
#     """

#     ####################################################
#     # Safety
#     ####################################################

#     if len(candidate_keyframes) <= 1:
#         return 1.0

#     ####################################################
#     # 1. Gap Consistency
#     ####################################################

#     gaps = np.diff(candidate_keyframes)

#     gap_std = float(np.std(gaps))

#     gap_consistency = 1.0 / (1.0 + gap_std)

#     ####################################################
#     # 2. Score Consistency
#     ####################################################

#     score_std = float(np.std(candidate_scores))

#     score_consistency = 1.0 / (1.0 + score_std)

#     ####################################################
#     # 3. Rank Consistency
#     ####################################################

#     rank_std = float(np.std(candidate_ranks))

#     rank_consistency = 1.0 / (1.0 + rank_std)

#     ####################################################
#     # Final
#     ####################################################

#     consistency = (
#         gap_weight * gap_consistency
#         + score_weight * score_consistency
#         + rank_weight * rank_consistency
#     )

#     return float(consistency)

import numpy as np

def compute_temporal_consistency(candidate_keyframes):
    """
    Temporal consistency chỉ đo độ đều của khoảng cách keyframe.

    Returns
    -------
    float in [0,1]
    """

    if len(candidate_keyframes) <= 1:
        return 1.0

    gaps = np.diff(candidate_keyframes)

    gap_std = float(np.std(gaps))

    return 1.0 / (1.0 + gap_std)

def temporal_sequence_reranking(
    temporal_results,
    config_path,
    config_name="reranking_temporal.yaml"
):
    """
    Re-rank beam search results.

    Parameters
    ----------
    temporal_results :
        output của temporal_sequence_retrieval()

    config_path :
        thư mục configs

    Returns
    -------
    reranked_results
    """

    if not temporal_results:
        return []

    # cfg = load_config(
    #     config_name,
    #     config_path
    # )
    if os.path.isfile(config_path):
        base_dir = os.path.dirname(config_path)
    else:
        base_dir = config_path
    full_config_path = os.path.join(base_dir, config_name)
    cfg = load_config(full_config_path)
    w = cfg["weights"]

    ##########################################################
    # chuẩn bị feature toàn cục
    ##########################################################

    avg_scores = [
        mean(seq["candidate_scores"])
        for seq in temporal_results
    ]


    beam_margins = [
        seq["beam_margin"]
        for seq in temporal_results
    ]


    avg_gaps = []
    for seq in temporal_results:
        keyframes = seq["candidate_keyframes"]
        if len(keyframes) > 1:
            gaps = np.diff(keyframes)
            avg_gaps.append(float(mean(gaps)))
        else:
            avg_gaps.append(0.0)

    ##########################################################
    # normalize
    ##########################################################
    norm_gaps = normalize(avg_gaps)
    norm_gaps = [
        1.0 - x
        for x in norm_gaps
    ]
    norm_scores = normalize(avg_scores)

    # rank càng nhỏ càng tốt
    # norm_ranks = normalize(avg_ranks)
    # norm_ranks = [
    #     1.0 - x
    #     for x in norm_ranks
    # ]

    norm_margins = normalize(beam_margins)

    ##########################################################
    # tính final score
    ##########################################################

    reranked = []

    for idx, seq in enumerate(temporal_results):

        temporal_consistency = compute_temporal_consistency(
            candidate_keyframes=seq["candidate_keyframes"]
        )
        keyframes = seq["candidate_keyframes"]
        avg_gap = float(mean(np.diff(keyframes))) if len(keyframes) > 1 else 0.0
        # final_score = (
        #     w.get("score", 0.55) * norm_scores[idx]
        #     # + w.get("rank", 0.10) * norm_ranks[idx]
        #     + w.get("beam_margin", 0.10) * norm_margins[idx]
        #     + w.get("temporal_gap", 0.15) * norm_gaps[idx]
        #     + w.get("temporal_consistency", 0.20) * temporal_consistency
        # )
        final_score = (
            w.get("semantic_score", 0.50) * norm_scores[idx]
            +
            w.get("beam_margin", 0.15) * norm_margins[idx]
            +
            w.get("temporal_gap", 0.15) * norm_gaps[idx]
            +
            w.get("temporal_consistency", 0.20) * temporal_consistency
        )
        new_seq = seq.copy()

        new_seq["rerank_score"] = final_score

        new_seq["rerank_features"] = {
            "avg_score": avg_scores[idx],
            # "avg_rank": avg_ranks[idx],
            "beam_margin": beam_margins[idx],
            "avg_gap": avg_gaps[idx],
            "temporal_consistency": temporal_consistency
        }

        reranked.append(new_seq)

    ##########################################################

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    for rank, seq in enumerate(reranked, start=1):
        seq["rerank_rank"] = rank
    return reranked