# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert GR00T N1.7 checkpoints between LeRobot and native RLinf layouts."""

from __future__ import annotations

import argparse
import pathlib

import torch

from rlinf.utils.ckpt_convertor.fsdp_convertor.utils import (
    save_state_dict_sharded_safetensors,
)
from rlinf.utils.ckpt_convertor.gr00t_n1d7._core import (
    build_native_processor_files,
    copy_metadata_tree,
    lerobot_to_native_state_dict,
    load_rlinf_state_dict,
    load_safetensors_checkpoint,
    native_to_lerobot_state_dict,
    state_dict_digest,
    write_json,
)

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "preserve": None,
}


def _ensure_empty_output(path: pathlib.Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def import_lerobot(args: argparse.Namespace) -> None:
    source = pathlib.Path(args.input_model)
    native_reference = pathlib.Path(args.native_reference)
    output = pathlib.Path(args.output_model)
    _ensure_empty_output(output)

    source_state = load_safetensors_checkpoint(source)
    native_state = lerobot_to_native_state_dict(source_state, dtype=_DTYPES[args.dtype])
    copy_metadata_tree(native_reference, output)
    processor, statistics, embodiment_ids = build_native_processor_files(
        source, native_reference
    )
    write_json(output / "processor_config.json", processor)
    write_json(output / "statistics.json", statistics)
    write_json(output / "embodiment_id.json", embodiment_ids)
    shard_count, total_size = save_state_dict_sharded_safetensors(
        native_state,
        str(output),
        max_shard_size=args.max_shard_size_gib * 1024**3,
    )
    write_json(
        output / "conversion_manifest.json",
        {
            "format": "rlinf-gr00t-n1.7",
            "source_format": "lerobot-groot-n1.7",
            "source": str(source.resolve()),
            "native_reference": str(native_reference.resolve()),
            "tensor_count": len(native_state),
            "tensor_contract_sha256": state_dict_digest(native_state),
            "shard_count": shard_count,
            "total_size": total_size,
        },
    )


def export_lerobot(args: argparse.Namespace) -> None:
    source = pathlib.Path(args.input_model)
    reference = pathlib.Path(args.lerobot_reference)
    output = pathlib.Path(args.output_model)
    _ensure_empty_output(output)

    native_state = load_rlinf_state_dict(source)
    reference_state = load_safetensors_checkpoint(reference)
    deployment_state = native_to_lerobot_state_dict(
        native_state, reference_state, dtype=_DTYPES[args.dtype]
    )
    copy_metadata_tree(reference, output)
    import safetensors.torch

    safetensors.torch.save_file(
        deployment_state,
        str(output / "model.safetensors"),
        metadata={"format": "pt"},
    )
    write_json(
        output / "conversion_manifest.json",
        {
            "format": "lerobot-groot-n1.7",
            "source_format": "rlinf-gr00t-n1.7",
            "source": str(source.resolve()),
            "lerobot_reference": str(reference.resolve()),
            "tensor_count": len(deployment_state),
            "tensor_contract_sha256": state_dict_digest(deployment_state),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    importer = subparsers.add_parser(
        "lerobot_to_rlinf", help="Convert a LeRobot GR00T N1.7 checkpoint for RLinf"
    )
    importer.add_argument("--input-model", required=True)
    importer.add_argument("--native-reference", required=True)
    importer.add_argument("--output-model", required=True)
    importer.add_argument("--dtype", choices=_DTYPES, default="bf16")
    importer.add_argument("--max-shard-size-gib", type=float, default=4.0)
    importer.set_defaults(run=import_lerobot)

    exporter = subparsers.add_parser(
        "rlinf_to_lerobot", help="Export an RLinf GR00T N1.7 checkpoint for LeRobot"
    )
    exporter.add_argument("--input-model", required=True)
    exporter.add_argument("--lerobot-reference", required=True)
    exporter.add_argument("--output-model", required=True)
    exporter.add_argument("--dtype", choices=_DTYPES, default="bf16")
    exporter.set_defaults(run=export_lerobot)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
