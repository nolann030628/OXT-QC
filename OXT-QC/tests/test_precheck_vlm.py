import numpy as np

from precheck_vlm import _image_like_to_uint8_rgb, render_tactile_panel


def test_image_like_to_uint8_rgb_accepts_hwc():
    frame = np.zeros((8, 6, 3), dtype=np.uint8)
    frame[..., 1] = 127

    out = _image_like_to_uint8_rgb(frame)

    assert out.shape == (8, 6, 3)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 1]) == 127


def test_image_like_to_uint8_rgb_accepts_chw():
    frame = np.zeros((3, 8, 6), dtype=np.uint8)
    frame[2, :, :] = 255

    out = _image_like_to_uint8_rgb(frame)

    assert out.shape == (8, 6, 3)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 2]) == 255


def test_render_tactile_panel_tiles_multiple_image_frames():
    frame = np.zeros((2, 6, 4, 3), dtype=np.uint8)
    frame[0, :, :, 0] = 255
    frame[1, :, :, 1] = 255

    panel = render_tactile_panel(
        frame=frame,
        tactile_kind="image",
        energy_series=np.array([0.0, 1.0], dtype=np.float32),
        t=0,
        out_size=(32, 16),
        vmin=0.0,
        vmax=1.0,
    )

    assert panel.shape == (16, 32, 3)
    assert panel.dtype == np.uint8
    assert np.any(panel[..., 0] > 0)
    assert np.any(panel[..., 1] > 0)
