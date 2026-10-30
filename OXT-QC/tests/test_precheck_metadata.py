import json
import numpy as np
import pytest
import zarr

from common.replay_buffer import ReplayBuffer
from precheck_metadata import check_dataset, main


@pytest.mark.parametrize("compressors", ["disk", "default", None])
def test_dict_export_reports_stored_image_compression(tmp_path, compressors):
    dataset_path = tmp_path / "example.zarr"
    buffer = ReplayBuffer.create_from_path(str(dataset_path), mode="a")
    episode = {
        "camera_ego_rgb": np.zeros((4, 8, 8, 3), dtype=np.uint8),
        "camera_ego_depth": np.zeros((4, 8, 8), dtype=np.uint16),
        "tactile_image": np.zeros((4, 8, 8), dtype=np.uint8),
        "right_tactile_data_tip": np.zeros((4, 1, 8, 8, 3), dtype=np.uint8),
        "timestamps": np.arange(4, dtype=np.float64) / 30,
    }
    buffer.add_episode(episode, compressors=compressors)
    output_dir = tmp_path / "output"

    main([str(dataset_path), "--output-dir", str(output_dir), "--no-metadata"])

    result = json.loads((output_dir / "dict.json").read_text())
    compressor = ReplayBuffer.resolve_compressor(compressors)
    expected = compressor.get_config() if compressor is not None else None
    for key in episode:
        assert result[key]["shape"] == list(episode[key].shape)
        assert result[key]["dtype"] == str(episode[key].dtype)
        if key == "timestamps":
            assert "compression" not in result[key]
        else:
            assert result[key]["compression"] == expected
    assert "compression" not in result["meta/episode_ends"]


@pytest.mark.parametrize("compressed", [True, False])
def test_dict_export_zarr_v3_image_compression(tmp_path, compressed):
    dataset_path = tmp_path / "v3.zarr"
    root = zarr.open_group(str(dataset_path), mode="w", zarr_format=3)
    data = root.create_group("data")
    compressors = [zarr.codecs.ZstdCodec(level=5)] if compressed else []
    image = data.create_array(
        "camera_ego_rgb", data=np.zeros((4, 8, 8, 3), dtype=np.uint8),
        compressors=compressors,
    )

    _, summary = check_dataset(dataset_path)

    assert summary["dict_export"]["camera_ego_rgb"]["compression"] == [
        codec.to_dict() for codec in image.compressors
    ]
