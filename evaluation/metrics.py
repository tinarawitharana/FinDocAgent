"""Shared scoring metrics used across all evaluation scripts: ANLS (Average Normalized
Levenshtein Similarity, the standard DocVQA metric) and set-based F1."""

def anls_score(prediction, ground_truth, threshold=0.5):
    """Normalized edit-distance similarity in [0, 1]; scores below `threshold` are
    clamped to 0, per the standard ANLS definition."""

    if not prediction or not ground_truth:
        return 0.0

    prediction = str(prediction).lower().strip()
    ground_truth = str(ground_truth).lower().strip()

    #levenstein distance
    dist = levenshtein_distance(prediction, ground_truth)
    max_len = max(len(prediction), len(ground_truth))

    if max_len == 0:
        return 1.0

    similarity = 1 - (dist / max_len)

    if similarity >= threshold:
        return similarity
    else:
        return 0.0

def levenshtein_distance(s1, s2):
    """Standard dynamic-programming edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] =dp[i-1][j-1]

            else:
                dp[i][j] = 1+ min(
                    dp[i-1][j],
                    dp[i][j-1],
                    dp[i-1][j-1]

                )

    return dp[m][n]

def calculate_f1(predicted_items, ground_truth_items):
    """Set-based F1 (case-insensitive exact match) between predicted and ground-truth items,
    used for line-item extraction where order doesn't matter."""

    if not predicted_items and not ground_truth_items:
        return 1.0
    if not predicted_items or not ground_truth_items:
        return 0.0

    pred_set = set(str(x).lower() for x in predicted_items)
    gt_set = set(str(x).lower() for x in ground_truth_items)

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gt_set) if gt_set else 0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision+recall)