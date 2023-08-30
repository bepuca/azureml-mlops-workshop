from common.evaluation.metrics import get_evaluation_metrics


def test_get_evaluation_metrics():
    true_labels = [0, 0, 1, 1]
    pred_labels = [0, 1, 1, 1]

    expected_metrics = {"accuracy": 0.75}

    metrics = get_evaluation_metrics(true_labels, pred_labels)

    assert metrics == expected_metrics
