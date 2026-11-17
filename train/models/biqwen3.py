"""BiQwen3: single-vector bi-encoder wrapping Qwen3VLModel.

Adapted from colpali-engine's BiQwen3 with key_mapping replaced by manual
state_dict remapping for compatibility with transformers <5.0.
"""

import re
from typing import ClassVar, Literal

import torch
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel

# Weight key mappings: Qwen3-VL-Embedding checkpoints store weights under
# "model." prefix (from Qwen3VLForConditionalGeneration), but Qwen3VLModel
# expects them without that prefix.
_KEY_MAPPINGS = [
    (re.compile(r"^model\.visual"), "visual"),
    (re.compile(r"^model\.language_model"), "language_model"),
    (re.compile(r"^model\."), ""),
]


def _remap_keys(state_dict):
    """Remap checkpoint keys from ConditionalGeneration to bare Model format."""
    new_sd = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in _KEY_MAPPINGS:
            if pattern.search(new_key):
                new_key = pattern.sub(replacement, new_key)
                break
        # Skip lm_head and other keys not in Qwen3VLModel
        if new_key.startswith("lm_head"):
            continue
        new_sd[new_key] = value
    return new_sd


class BiQwen3(Qwen3VLModel):
    """Single-vector bi-encoder with last-token pooling + L2 normalization."""

    main_input_name: ClassVar[str] = "doc_input_ids"

    def __init__(self, config: Qwen3VLConfig, **kwargs):
        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", None))
        attn_impl = kwargs.pop("attn_implementation", None)
        use_cache = kwargs.pop("use_cache", None)

        super().__init__(config=config)
        self.padding_side = "left"
        self.post_init()

        if dtype is not None:
            self.to(dtype=dtype)
        if use_cache is not None:
            self.config.use_cache = use_cache
        if attn_impl is not None and hasattr(self, "set_attn_implementation"):
            self.set_attn_implementation(attn_impl)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # For transformers <5.0: handle key remapping manually
        import inspect

        from transformers import PreTrainedModel

        sig = inspect.signature(PreTrainedModel.from_pretrained)
        if "key_mapping" in sig.parameters:
            # transformers >=5.0: use native key_mapping
            kwargs.setdefault(
                "key_mapping",
                {
                    r"^model\.visual": "visual",
                    r"^model\.language_model": "language_model",
                    r"^model\.": "",
                },
            )
            return super().from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )

        # transformers <5.0: load with remapped state_dict
        import glob
        from pathlib import Path

        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoConfig

        kwargs.get("dtype", kwargs.get("torch_dtype", None))

        # Resolve model path
        model_path = pretrained_model_name_or_path
        if not Path(model_path).exists():
            model_path = snapshot_download(pretrained_model_name_or_path)

        # Load config
        config = AutoConfig.from_pretrained(model_path)
        model = cls(
            config,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("dtype", "torch_dtype", "attn_implementation", "use_cache")
            },
        )

        # Load and remap state dict
        safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
        if safetensor_files:
            state_dict = {}
            for f in safetensor_files:
                state_dict.update(load_file(f))
        else:
            bin_files = sorted(glob.glob(str(Path(model_path) / "*.bin")))
            state_dict = {}
            for f in bin_files:
                state_dict.update(torch.load(f, map_location="cpu", weights_only=True))

        state_dict = _remap_keys(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(
                f"BiQwen3: {len(missing)} missing keys (expected for embedding-only model)"
            )
        if unexpected:
            print(f"BiQwen3: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

        return model

    def forward(
        self,
        pooling_strategy: Literal["cls", "last", "mean"] = "last",
        bidirectional: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if "pixel_values" in kwargs and kwargs["pixel_values"].dim() == 3:
            # ColQwen3Processor pads pixel_values to (batch, max_patches, dim);
            # undo padding back to flat (total_patches, dim) for Qwen3VLModel.
            offsets = kwargs["image_grid_thw"].prod(dim=1).tolist()
            kwargs["pixel_values"] = torch.cat(
                [
                    pixel_sequence[:offset]
                    for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)
                ],
                dim=0,
            )
        # Standard Qwen3VLProcessor already gives flat (total_patches, dim) — no-op.
        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        if bidirectional and "attention_mask" in kwargs:
            # Convert 2D padding mask (batch, seq_len) to 4D bidirectional mask.
            # 4D masks bypass create_causal_mask in transformers and go straight
            # to the attention layers, effectively disabling causal masking.
            orig_mask = kwargs["attention_mask"]  # (batch, seq_len), 1=valid, 0=pad
            if orig_mask.ndim == 2:
                batch, seq_len = orig_mask.shape
                # Build (batch, 1, seq_len, seq_len): 0.0=attend, large_neg=mask
                # Rows: which positions are querying. Cols: which positions to attend to.
                # We mask columns where padding exists.
                mask_4d = (
                    orig_mask[:, None, None, :]
                    .expand(batch, 1, seq_len, seq_len)
                    .to(torch.bfloat16)
                )
                mask_4d = (1.0 - mask_4d) * torch.finfo(torch.bfloat16).min
                kwargs["attention_mask"] = mask_4d

        last_hidden_states = (
            super()
            .forward(
                *args,
                **kwargs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            .last_hidden_state
        )

        if pooling_strategy == "cls":
            pooled = last_hidden_states[:, 0]
        elif pooling_strategy == "last":
            pooled = last_hidden_states[:, -1]
        elif pooling_strategy == "mean":
            mask = kwargs["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

        return pooled / pooled.norm(dim=-1, keepdim=True)

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
