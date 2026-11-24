import pytest

from local_jev.calibration import Bucket, Record, brier, ece, reliability


def test_reliability_groups_records_by_bucket_and_skips_empty_ones():
    records = [Record(0.95, True), Record(0.91, False), Record(0.55, True)]
    assert reliability(records) == [
        Bucket(low=0.5, high=0.6, items=1, mean_confidence=0.55, accuracy=1.0),
        Bucket(low=0.9, high=1.0, items=2, mean_confidence=0.93, accuracy=0.5),
    ]


def test_a_confidence_of_one_lands_in_the_last_bucket():
    assert reliability([Record(1.0, True)]) == [Bucket(0.9, 1.0, 1, 1.0, 1.0)]


def test_a_confidence_on_a_bucket_edge_starts_that_bucket():
    assert reliability([Record(0.7, True)])[0].low == 0.7


def test_the_table_starts_at_the_lowest_confidence_seen():
    assert reliability([Record(0.35, False)])[0] == Bucket(0.3, 0.4, 1, 0.35, 0.0)


def test_gap_is_accuracy_minus_mean_confidence():
    assert Bucket(0.9, 1.0, 2, 0.93, 0.5).gap == pytest.approx(-0.43)


def test_ece_weights_each_bucket_gap_by_its_share_of_the_records():
    records = [Record(0.95, True), Record(0.91, False), Record(0.55, True)]
    # bucket 0.5: 1/3 * |1.0 - 0.55|; bucket 0.9: 2/3 * |0.5 - 0.93|
    assert ece(records) == pytest.approx(1 / 3 * 0.45 + 2 / 3 * 0.43)


def test_ece_uses_exact_bucket_values_not_the_rounded_table():
    records = [Record(0.9, True), Record(0.9, True), Record(0.9, False)]
    assert ece(records) == pytest.approx(abs(2 / 3 - 0.9))


def test_brier_is_the_mean_squared_error_against_a_right_answer():
    assert brier([Record(0.8, True), Record(0.6, False)]) == pytest.approx((0.04 + 0.36) / 2)


@pytest.mark.parametrize("function", [reliability, ece, brier])
def test_an_empty_record_list_raises(function):
    with pytest.raises(ValueError, match="no records"):
        function([])
