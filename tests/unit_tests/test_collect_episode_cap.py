"""Collection-suite episode-cap regression tests."""

from __future__ import annotations

from concurrent.futures import Future

import gymnasium as gym
import numpy as np

from rlinf.envs.wrappers.collect_episode import CollectEpisode


class _DummyEnv(gym.Env):
    def reset(self, *, seed=None, options=None):
        del seed, options
        return {"state": np.zeros(6, dtype=np.float32)}, {}


def test_collection_stops_recording_after_per_env_cap(tmp_path, monkeypatch):
    wrapper = CollectEpisode(
        _DummyEnv(),
        save_dir=str(tmp_path),
        num_envs=1,
        max_episodes_per_env=1,
    )
    monkeypatch.setattr(wrapper, "_flush_episode", lambda *_: None)
    wrapper.reset()
    wrapper._record_step(
        np.ones((1, 6), dtype=np.float32),
        {"state": np.ones((1, 6), dtype=np.float32)},
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=bool),
        np.zeros(1, dtype=bool),
        {},
    )
    wrapper._maybe_flush(np.ones(1, dtype=bool), np.zeros(1, dtype=bool))

    assert wrapper._completed_episodes == [1]
    assert wrapper._buffers[0]["actions"] == []

    wrapper._record_step(
        np.full((1, 6), 2, dtype=np.float32),
        {"state": np.full((1, 6), 2, dtype=np.float32)},
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=bool),
        np.zeros(1, dtype=bool),
        {},
    )
    assert wrapper._buffers[0]["actions"] == []
    wrapper.close()


def test_collection_rejects_non_positive_episode_cap(tmp_path):
    try:
        CollectEpisode(
            _DummyEnv(),
            save_dir=str(tmp_path),
            max_episodes_per_env=0,
        )
    except ValueError as exc:
        assert "max_episodes_per_env" in str(exc)
    else:
        raise AssertionError("expected a non-positive episode cap to fail")


def test_close_accepts_periodically_finalized_lerobot_writer(tmp_path):
    class _FinalizedWriter:
        dataset = None

        def finalize(self):
            raise AssertionError("an already-finalized writer must not finalize twice")

    wrapper = CollectEpisode(
        _DummyEnv(),
        save_dir=str(tmp_path),
        export_format="lerobot",
    )
    wrapper._lerobot_writer = _FinalizedWriter()
    wrapper.close()

    assert wrapper._lerobot_writer is None


def test_finalize_collection_drains_and_finalizes_without_closing(tmp_path):
    class _Writer:
        dataset = object()

        def __init__(self):
            self.finalized = False

        def finalize(self):
            self.finalized = True

    wrapper = CollectEpisode(
        _DummyEnv(),
        save_dir=str(tmp_path),
        export_format="lerobot",
    )
    writer = _Writer()
    wrapper._lerobot_writer = writer
    completed = Future()
    completed.set_result(None)
    wrapper._futures = [completed]

    wrapper.finalize_collection()

    assert writer.finalized
    assert wrapper._lerobot_writer is None
    assert wrapper._futures == []
    assert wrapper._executor is not None
    assert not wrapper._closed
    wrapper.close()
