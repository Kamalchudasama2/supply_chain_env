def compute_score(total_reward: float, max_possible: float):
    score = total_reward / max_possible
    return max(0.0, min(score, 1.0))