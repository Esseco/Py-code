import pytest

from Process_MLIP.forgetting import assess_forgetting


def test_fraction_uses_distinct_structures_and_includes_ten_percent():
    ids = [str(i) for i in range(20)]
    result = assess_forgetting(ids + [ids[0]], [ids[0], ids[0], ids[1]], coverage_complete=True)
    assert result['collapse_count'] == 2
    assert result['structure_count'] == 20
    assert result['direct_sampling_required'] is True
    assert result['reliability_review_required'] is True
    assert assess_forgetting(ids + ['20'], [ids[0], ids[1]], coverage_complete=True)['direct_sampling_required'] is False


def test_missing_or_foreign_structure_identity_is_rejected():
    with pytest.raises(ValueError):
        assess_forgetting([], [], coverage_complete=True)
    with pytest.raises(ValueError):
        assess_forgetting(['one'], ['other'], coverage_complete=True)


def test_incomplete_round_is_not_cleared_or_automatically_sampled():
    result = assess_forgetting(['one', 'two'], [], coverage_complete=False)
    assert result['reliability_review_required'] is True
    assert result['direct_sampling_required'] is None
