from typing import Sequence

from sklearn.metrics import accuracy_score


def get_evaluation_metrics(
    true_labels: Sequence[float], pred_labels: Sequence[float]
) -> dict[str, float]:
    return {"accuracy": round(accuracy_score(true_labels, pred_labels), 4)}
