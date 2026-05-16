import re
import numpy as np
from collections import Counter
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, mean_squared_error


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def regression_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    pearson = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else 0.0
    spearman = float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else 0.0
    return {
        "mse": mse,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
    }


def token_classification_metrics(y_true, y_pred):
    total_correct = 0
    total_tokens = 0
    
    for true_tags, pred_tags in zip(y_true, y_pred):
        # Chuyển sang uppercase để so sánh
        true_list = str(true_tags).upper().split()
        pred_list = str(pred_tags).upper().split()
        
        min_len = min(len(true_list), len(pred_list))
        for i in range(min_len):
            if true_list[i] == pred_list[i]:
                total_correct += 1
            total_tokens += 1
        
        # Tính tokens bị thiếu hoặc thừa
        total_tokens += abs(len(true_list) - len(pred_list))
    
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return {
        "token_accuracy": accuracy,
        "correct_tokens": total_correct,
        "total_tokens": total_tokens,
    }


def _normalize_answer(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text


def _f1_score(prediction, ground_truth):
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def qa_metrics(y_true, y_pred):
    total = len(y_true)
    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0}

    em_sum = 0.0
    f1_sum = 0.0
    for answers, pred in zip(y_true, y_pred):
        if not answers:
            continue
        em = max(_exact_match(pred, a) for a in answers)
        f1 = max(_f1_score(pred, a) for a in answers)
        em_sum += 1.0 if em else 0.0
        f1_sum += f1

    return {
        "exact_match": em_sum / total,
        "f1": f1_sum / total,
    }