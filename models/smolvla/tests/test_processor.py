import numpy as np

from smolvla_mlx.processor import PolicyProcessor, pad_vector, resize_with_pad


def _processor() -> PolicyProcessor:
    return PolicyProcessor(
        {"steps": [{"registry_name": "rename_observations_processor", "config": {"rename_map": {"observation.images.wrist": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}}}]},
        {"steps": []},
    )


def test_resize_with_pad_preserves_chw_and_range():
    source = np.full((480, 640, 3), 255, dtype=np.uint8)
    result = resize_with_pad(source)
    assert result.shape == (3, 512, 512)
    assert result.max() == 1.0
    assert np.all(result[:, :128, :] == 0.0)
    assert np.all(result[:, 128:, :] == 1.0)


def test_state_action_padding():
    padded = pad_vector(np.arange(6), 32)
    assert padded.shape == (32,)
    np.testing.assert_array_equal(padded[:6], np.arange(6))
    assert np.count_nonzero(padded[6:]) == 0


def test_action_mean_std_roundtrip():
    processor = PolicyProcessor(
        {"steps": []},
        {"steps": []},
        unnormalizer_state={
            "action.mean": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            "action.std": np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        },
    )
    normalized = np.array([[-0.5, 0.0, 0.25, 1.0, -1.0, 2.0]], dtype=np.float32)
    restored = processor.normalize_action(processor.unnormalize_action(normalized))
    np.testing.assert_allclose(restored, normalized, atol=1e-6)


def test_camera_rename_map_is_checkpoint_contract():
    assert _processor().rename_map == {
        "observation.images.wrist": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
    }
