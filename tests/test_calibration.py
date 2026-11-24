import pytest

from local_jev.calibration import (
    Bucket,
    Record,
    accuracy,
    brier,
    ece,
    mean_confidence,
    reliability,
)


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


@pytest.mark.parametrize("function", [reliability, ece, brier, mean_confidence, accuracy])
def test_an_empty_record_list_raises(function):
    with pytest.raises(ValueError, match="no records"):
        function([])


@pytest.mark.parametrize("width", [0, 0.3, -0.1, 1.5])
@pytest.mark.parametrize("function", [reliability, ece])
def test_a_width_that_does_not_split_one_into_whole_buckets_raises(function, width):
    with pytest.raises(ValueError, match=f"width {width}"):
        function([Record(0.5, True)], width=width)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
@pytest.mark.parametrize("function", [reliability, ece, brier, mean_confidence, accuracy])
def test_a_confidence_outside_zero_to_one_raises(function, confidence):
    with pytest.raises(ValueError, match=f"confidence {confidence}"):
        function([Record(0.5, True), Record(confidence, False)])


def test_a_quarter_width_puts_three_quarters_and_one_in_the_last_bucket():
    records = [Record(0.75, True), Record(1.0, False)]
    assert reliability(records, width=0.25) == [Bucket(0.75, 1.0, 2, 0.875, 0.5)]


def test_six_tenths_starts_the_six_tenths_bucket_despite_float_error():
    assert reliability([Record(0.6, True)]) == [Bucket(0.6, 0.7, 1, 0.6, 1.0)]


def test_mean_confidence_and_accuracy_are_exact():
    records = [Record(0.9, True), Record(0.6, False), Record(0.75, True)]
    assert mean_confidence(records) == pytest.approx(0.75)
    assert accuracy(records) == pytest.approx(2 / 3)
