from typing import List
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


# https://github.com/yangkevin2/coronavirus_data/blob/master/scripts/evaluate_auc.py
def prc_auc_score(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.
    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def evaluate_auc(targets: List[int], preds: List[float]) -> (float, float):
    roc_auc = roc_auc_score(targets, preds)
    prc_auc = prc_auc_score(targets, preds)
    return roc_auc, prc_auc

