"""RECAP CFG dataset-schema contracts."""

from rlinf.workers.sft.fsdp_cfg_worker import _resolve_action_sequence_keys


def test_cfg_action_key_accepts_canonical_so101_schema():
    assert _resolve_action_sequence_keys(("action",), {"action": {}}) == ("action",)


def test_cfg_action_key_accepts_rlinf_collection_schema():
    assert _resolve_action_sequence_keys(("action",), {"actions": {}}) == (
        "actions",
    )
