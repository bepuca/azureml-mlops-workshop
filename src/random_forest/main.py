from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from func_to_script import script
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from transform import prepare_sample

from common.core.constants import DATASET_FILENAME, DataFields, DataSets
from common.evaluation.metrics import get_evaluation_metrics
from common.logging.azureml_logger import AzureMLLogger


def load_dataset_from_df(df: pd.DataFrame, images_path: Path) -> tuple[np.array, np.array]:
    X = np.stack(
        [prepare_sample(Image.open(images_path / filepath)) for filepath in df[DataFields.FILEPATH]]
    )
    y = df[DataFields.LABEL].values
    return X, y


@script
def main(
    images_path: str,
    splits_path: str,
    model_save_path: str = "./outputs",
    n_estimators: int = 4,
    max_depth: int = 2,
    max_features: int = 2,
):
    logger = AzureMLLogger()
    logger.set_tags(
        {"n_estimators": n_estimators, "max_depth": max_depth, "max_features": max_features}
    )

    images_path = Path(images_path)
    splits_path = Path(splits_path)

    model_save_path = Path(model_save_path)
    model_save_path.mkdir(parents=True, exist_ok=True)
    model_filepath = model_save_path / "model.pkl"

    splits_df = pd.read_csv(splits_path / DATASET_FILENAME)

    X_train, y_train = load_dataset_from_df(
        splits_df.query(f"{DataFields.SET} == '{DataSets.TRAIN}'"), images_path
    )
    X_eval, y_eval = load_dataset_from_df(
        splits_df.query(f"{DataFields.SET} == '{DataSets.EVAL}'"), images_path
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, model_filepath)

    y_train_pred = model.predict(X_train)
    y_eval_pred = model.predict(X_eval)

    # This is our evaluation framework for now
    train_metrics = get_evaluation_metrics(y_train, y_train_pred)
    eval_metrics = get_evaluation_metrics(y_eval, y_eval_pred)

    metrics = {
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"eval_{k}": v for k, v in eval_metrics.items()},
    }
    logger.log_metrics(metrics)


if __name__ == "__main__":
    # No arguments passed because we leverage the @script decorator
    main()
