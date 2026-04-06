import numpy as np

from mock_lab.spectroscopy.tips import get_co_partition_sum, get_partition_sum


def test_get_co_partition_sum_matches_local_tips_reference_value() -> None:
    assert np.isclose(get_co_partition_sum(296.0), 107.42051)


def test_get_partition_sum_linearly_interpolates_between_integer_temperatures() -> None:
    q_296 = get_partition_sum("CO", "1", 296.0)
    q_297 = get_partition_sum("CO", "1", 297.0)
    q_296p5 = get_partition_sum("CO", "1", 296.5)

    assert np.isclose(q_296p5, 0.5 * (q_296 + q_297))
