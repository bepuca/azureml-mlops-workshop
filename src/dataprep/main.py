from pathlib import Path
from typing import Iterable

import pandas as pd
from func_to_script import script

from common.core.constants import DATASET_FILENAME, DataFields, DataSets
from common.logging.azureml_logger import AzureMLLogger


def split_into_sets(paths: list[Path], n_eval_samples: int, n_test_samples: int):
    """Split `paths` into train, eval and test."""
    # We only need the relative path for each file (e.g. 3/108.png)
    paths = ["/".join(p.parts[-2:]) for p in paths]
    paths = sorted(paths)

    eval_set = paths[:n_eval_samples]
    test_set = paths[n_eval_samples : n_eval_samples + n_test_samples]
    train_set = paths[n_eval_samples + n_test_samples :]

    return train_set, eval_set, test_set


def build_splits_dataframe(
    filepaths_3s: Iterable, filepaths_7s: Iterable, n_eval_samples: int, n_test_samples
) -> pd.DataFrame:
    n_3s = len(filepaths_3s)
    n_7s = len(filepaths_7s)
    n_tot = n_3s + n_7s

    if n_tot <= (n_eval_samples + n_test_samples):
        raise ValueError(
            f"Total number of samples ({n_tot}) is not greater than the sum of "
            f"evaluation and test samples ({n_eval_samples} + {n_test_samples}). "
            "Not enough samples to create a train set."
        )

    fraction_3s = n_3s / n_tot
    train_3s, eval_3s, test_3s = split_into_sets(
        filepaths_3s, round(n_eval_samples * fraction_3s), round(n_test_samples * fraction_3s)
    )

    fraction_7s = n_7s / n_tot
    train_7s, eval_7s, test_7s = split_into_sets(
        filepaths_7s, round(n_eval_samples * fraction_7s), round(n_test_samples * fraction_7s)
    )

    rows = [
        {DataFields.FILEPATH: f, DataFields.LABEL: i, DataFields.SET: s}
        for i, sets in enumerate([(train_3s, eval_3s, test_3s), (train_7s, eval_7s, test_7s)])
        for s, filepaths in zip([DataSets.TRAIN, DataSets.EVAL, DataSets.TEST], sets)
        for f in filepaths
    ]

    return pd.DataFrame.from_records(rows)


@script
def main(
    images_path: str, output_path: str, n_eval_samples: int = 1000, n_test_samples: int = 1000
):
    """Find all images, deterministically split between sets and save them as CSV including labels

    :param images_path: Path where all images are stored.
    :param output_path: Path where to save the CSV with the sets and labels.
    :param n_eval_samples: Number of desired samples for the evaluation set.
    :param n_test_samples: Number of desired samples for the test set.
    """
    logger = AzureMLLogger()

    images_path = Path(images_path)
    output_path = Path(output_path)

    filepaths_3s = list(images_path.glob("3/*"))
    filepaths_7s = list(images_path.glob("7/*"))

    df = build_splits_dataframe(filepaths_3s, filepaths_7s, n_eval_samples, n_test_samples)
    df.to_csv(output_path / DATASET_FILENAME, index=False)

    logger.set_tags(
        {
            "n_train_samples": len(df.query(f"{DataFields.SET} == '{DataSets.TRAIN}'")),
            "n_eval_samples": n_eval_samples,
            "n_test_samples": n_test_samples,
        }
    )


if __name__ == "__main__":
    # No arguments passed because we leverage the @script decorator
    main()
