"""
Basic tests for ml_pipeline.py

Run with:  pytest
"""
import numpy as np
import pandas as pd
import pytest

from ml_pipeline import load_data, prepare_features, split_columns, get_preprocessor


def test_load_data_creates_binary_target():
    """load_data() must return a target column with exactly two classes: 0 and 1."""
    df = load_data()
    assert "target" in df.columns
    assert set(df["target"].unique()).issubset({0, 1})
    assert df["target"].nunique() == 2


def test_load_data_no_missing_rows():
    """The loaded dataframe should not silently drop all rows."""
    df = load_data()
    assert len(df) > 0


def test_prepare_features_excludes_grades_by_default():
    """G1, G2, G3 must NOT be feature columns (that would be data leakage)."""
    df = load_data()
    X, y = prepare_features(df, include_prior_grades=False)
    assert "G1" not in X.columns
    assert "G2" not in X.columns
    assert "G3" not in X.columns
    assert len(X) == len(y)


def test_prepare_features_can_include_prior_grades():
    """When explicitly requested, G1/G2 may be included (G3 is always excluded)."""
    df = load_data()
    X, y = prepare_features(df, include_prior_grades=True)
    assert "G3" not in X.columns  # G3 defines the target, never a feature
    # G1/G2 should be present if they exist in the raw dataset
    if "G1" in df.columns:
        assert "G1" in X.columns


def test_prepare_features_rejects_empty_result():
    """If dropping target/grade columns leaves nothing, it should raise, not silently continue."""
    df = pd.DataFrame({"target": [0, 1], "G3": [5, 15]})
    with pytest.raises(ValueError):
        prepare_features(df, include_prior_grades=False)


def test_split_columns_separates_types_correctly():
    df = load_data()
    X, _ = prepare_features(df)
    cat_cols, num_cols = split_columns(X)
    # Every column must land in exactly one bucket
    assert set(cat_cols) | set(num_cols) == set(X.columns)
    assert set(cat_cols) & set(num_cols) == set()


def test_get_preprocessor_transforms_without_error():
    """The preprocessor pipeline should fit/transform the real feature set cleanly."""
    df = load_data()
    X, _ = prepare_features(df)
    cat_cols, num_cols = split_columns(X)
    preprocessor = get_preprocessor(cat_cols, num_cols)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert not np.isnan(transformed).any()
