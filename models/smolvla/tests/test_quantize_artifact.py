import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from quantization.artifact import quantize_checkpoint


def test_w8a8_group_quantizes_width_divisible_by_64(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    weight = mx.ones((32, 64))
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {"model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj.weight": weight},
    )
    (source / "config.json").write_text("{}\n")
    stem = "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj"
    report = quantize_checkpoint(
        source,
        tmp_path / "dst",
        scheme="w8a8",
        activation_amax={stem: np.full(64, 4.0, dtype=np.float32)},
    )
    manifest = json.loads((tmp_path / "dst" / "quantization.json").read_text())
    assert report["quantized_layers"] == 1
    assert report["calibrated_layers"] == 1
    layer = manifest["layers"][stem]
    assert layer["rotate"] is True
    assert layer["weight_quant"] == "group"
    saved = mx.load(str(tmp_path / "dst" / "weights.safetensors"))
    assert stem + ".qweight" in saved
    assert stem + ".weight" not in saved
    assert not np.allclose(np.array(saved[stem + ".smooth_scale"]), 1.0)


def test_w8a8_keeps_per_row_codes_when_width_is_not_a_group(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {"model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight": mx.ones((8, 48))},
    )
    report = quantize_checkpoint(source, tmp_path / "dst", scheme="w8a8")
    stem = "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj"
    manifest = json.loads((tmp_path / "dst" / "quantization.json").read_text())
    assert report["quantized_layers"] == 1
    assert manifest["layers"][stem]["weight_quant"] == "row"
    assert manifest["layers"][stem]["rotate"] is False
    saved = mx.load(str(tmp_path / "dst" / "weights.safetensors"))
    assert stem + ".codes" in saved
    assert stem + ".weight" not in saved


def test_quantize_copies_processor_statistics(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {"model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj.weight": mx.zeros((32, 64))},
    )
    mx.save_safetensors(
        str(source / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"),
        {"action.mean": mx.zeros((6,)), "action.std": mx.ones((6,))},
    )
    (source / "policy_postprocessor.json").write_text("{}\n")
    quantize_checkpoint(source, tmp_path / "dst", scheme="w8a8")
    destination = tmp_path / "dst"
    assert (destination / "policy_postprocessor.json").is_file()
    assert (destination / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").is_file()
    assert not (destination / "model.safetensors").exists()


def test_mlx_w8_uses_group_affine_and_skips_expert_width(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj.weight": mx.ones((8, 64)),
            "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight": mx.ones((8, 720)),
        },
    )
    report = quantize_checkpoint(source, tmp_path / "dst", scheme="mlx-w8", group_size=64)
    manifest = json.loads((tmp_path / "dst" / "quantization.json").read_text())
    assert report["quantized_layers"] == 1
    assert report["skipped_layers"] == 1
    saved = mx.load(str(tmp_path / "dst" / "weights.safetensors"))
    assert "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj.qweight" in saved
    assert "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight" in saved
    assert manifest["layers"]["model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj"]["group_size"] == 64


def test_activation_hook_records_channel_max():
    import mlx.nn as nn

    from quantization.collect import _attach_amax

    layer = nn.Linear(4, 2)
    captured: list = []
    _attach_amax(layer, "proj", captured)
    layer(mx.array([[1.0, -3.0, 0.5, 0.0]]))
    mx.eval(captured[0][1])
    assert captured[0][0] == "proj"
    np.testing.assert_allclose(np.array(captured[0][1]), [1.0, 3.0, 0.5, 0.0])


def test_selective_w4_keeps_down_projection(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {"model.vlm_with_expert.lm_expert.layers.0.mlp.down_proj.weight": mx.zeros((16, 64))},
    )
    report = quantize_checkpoint(source, tmp_path / "dst", scheme="w4a4-selective")
    assert report["quantized_layers"] == 0
