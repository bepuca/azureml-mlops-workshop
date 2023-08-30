from pathlib import Path

import pandas as pd
import pytest
from main import build_splits_dataframe

from common.core.constants import DataFields, DataSets


def test_build_splits_dataframe_raises_when_insufficient_samples():
    filepaths_3s = range(10)
    filepaths_7s = range(20)
    n_eval_samples = 30
    n_test_samples = 50

    with pytest.raises(ValueError):
        build_splits_dataframe(filepaths_3s, filepaths_7s, n_eval_samples, n_test_samples)


def test_build_splits_dataframe():
    filepaths_3s = [Path(f"a/path/we/do/not/want/3/{i}.png") for i in [1, 4, 2, 5, 7, 6, 3, 8]]
    filepaths_7s = [Path(f"a/path/we/do/not/want/7/{i}.png") for i in range(4)]
    n_eval_samples = 3
    n_test_samples = 6

    expected_df = pd.DataFrame.from_records(
        [
            {DataFields.FILEPATH: "3/7.png", DataFields.LABEL: 0, DataFields.SET: DataSets.TRAIN},
            {DataFields.FILEPATH: "3/8.png", DataFields.LABEL: 0, DataFields.SET: DataSets.TRAIN},
            {DataFields.FILEPATH: "3/1.png", DataFields.LABEL: 0, DataFields.SET: DataSets.EVAL},
            {DataFields.FILEPATH: "3/2.png", DataFields.LABEL: 0, DataFields.SET: DataSets.EVAL},
            {DataFields.FILEPATH: "3/3.png", DataFields.LABEL: 0, DataFields.SET: DataSets.TEST},
            {DataFields.FILEPATH: "3/4.png", DataFields.LABEL: 0, DataFields.SET: DataSets.TEST},
            {DataFields.FILEPATH: "3/5.png", DataFields.LABEL: 0, DataFields.SET: DataSets.TEST},
            {DataFields.FILEPATH: "3/6.png", DataFields.LABEL: 0, DataFields.SET: DataSets.TEST},
            {DataFields.FILEPATH: "7/3.png", DataFields.LABEL: 1, DataFields.SET: DataSets.TRAIN},
            {DataFields.FILEPATH: "7/0.png", DataFields.LABEL: 1, DataFields.SET: DataSets.EVAL},
            {DataFields.FILEPATH: "7/1.png", DataFields.LABEL: 1, DataFields.SET: DataSets.TEST},
            {DataFields.FILEPATH: "7/2.png", DataFields.LABEL: 1, DataFields.SET: DataSets.TEST},
        ]
    )

    df = build_splits_dataframe(filepaths_3s, filepaths_7s, n_eval_samples, n_test_samples)

    pd.testing.assert_frame_equal(df, expected_df)
    assert len(df.query(f"{DataFields.SET} == '{DataSets.EVAL}'")) == n_eval_samples
    assert len(df.query(f"{DataFields.SET} == '{DataSets.TEST}'")) == n_test_samples
