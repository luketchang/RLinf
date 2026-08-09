# GR00T N1.7 checkpoint conversion

RLinf trains the native NVIDIA GR00T N1.7 module. LeRobot deploys the same
weights through `GrootPolicy`, whose state dict adds a `_groot_model.` wrapper
and whose processor metadata uses LeRobot's processor-pipeline format. These
converters make that boundary explicit and validate it strictly.

Import a LeRobot SFT checkpoint for RLinf evaluation or training:

```bash
python -m rlinf.utils.ckpt_convertor.gr00t_n1d7.convert lerobot_to_rlinf \
  --input-model /checkpoints/lerobot/pretrained_model \
  --native-reference /checkpoints/nvidia/GR00T-N1.7-3B \
  --output-model /checkpoints/rlinf
```

Export a consolidated RLinf checkpoint for `lerobot-rollout`:

```bash
python -m rlinf.utils.ckpt_convertor.gr00t_n1d7.convert rlinf_to_lerobot \
  --input-model /runs/step_000100 \
  --lerobot-reference /checkpoints/lerobot/pretrained_model \
  --output-model /checkpoints/lerobot-step-000100
```

The LeRobot reference supplies the deployment config and processor contract.
It must be the SFT checkpoint from which RL training started. The exporter
rejects partial loads, shape changes, and unexpected non-RL tensors. RL-only
value/exploration heads are intentionally omitted.
