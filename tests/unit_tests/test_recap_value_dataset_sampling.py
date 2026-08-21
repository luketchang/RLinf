from rlinf.data.datasets.recap.value_dataset import uniform_sample_indices


def test_uniform_sample_indices_spans_complete_range() -> None:
    assert uniform_sample_indices(list(range(100)), 5) == [0, 24, 49, 74, 99]


def test_uniform_sample_indices_preserves_small_inputs() -> None:
    assert uniform_sample_indices([2, 5, 9], 5) == [2, 5, 9]
