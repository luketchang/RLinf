from rlinf.utils.runner_utils import EarlyStopController


def test_early_stop_supports_maximized_custom_metric() -> None:
    controller = EarlyStopController(
        {
            "enabled": True,
            "patience": 2,
            "min_delta": 0.01,
            "monitor": "value_spearman",
            "mode": "max",
        }
    )

    assert controller.update({"value_spearman": 0.4}) == (False, True)
    assert controller.update({"value_spearman": 0.405}) == (False, False)
    assert controller.update({"value_spearman": 0.39}) == (True, False)
    assert controller.best_monitored_value == 0.4


def test_disabled_early_stop_still_reports_best_checkpoint() -> None:
    controller = EarlyStopController(
        {"enabled": False, "monitor": "value_spearman", "mode": "max"}
    )

    assert controller.update({"value_spearman": 0.2}) == (False, True)
    assert controller.update({"value_spearman": 0.1}) == (False, False)
