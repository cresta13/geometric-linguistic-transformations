"""Regression tests for the GLT-STEER token-direction audit protocol."""

from run_glt_steer_confirmatory_fixed_params import TRAIN_ROWS
from run_glt_steer_token_direction_controls import build_token_audit_training_pairs


def test_token_audit_uses_confirmatory_source_count_and_diversity():
    train_df = build_token_audit_training_pairs()
    sources = train_df["source"].drop_duplicates()

    assert len(sources) == TRAIN_ROWS
    assert len(train_df) == TRAIN_ROWS * 3
    assert sources.str.split().str[1].nunique() == 12


def test_token_audit_has_no_duplicate_training_sources():
    train_df = build_token_audit_training_pairs()

    assert train_df["source"].nunique() == TRAIN_ROWS
