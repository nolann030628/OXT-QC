import os
from pathlib import Path
import subprocess
import sys

import numcodecs
import numpy as np
import pytest

from common.replay_buffer import ReplayBuffer


def _run_python(script, *arguments):
    environment = os.environ.copy()
    project = str(Path(__file__).resolve().parents[1])
    environment['PYTHONPATH'] = os.pathsep.join(
        [project, environment.get('PYTHONPATH', '')]
    )
    result = subprocess.run(
        [sys.executable, '-c', script, *map(str, arguments)],
        env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.mark.parametrize(
    'codec_name, options, lossless',
    [
        ('Jpeg', {'level': 95}, False),
        ('Jpegxl', {'lossless': True}, True),
        ('Jpegxl', {'level': 99, 'lossless': False}, False),
        ('Jpeg2k', {'reversible': True}, True),
        ('Jpegls', {'level': 0}, True),
        ('Png', {}, True),
        ('Webp', {'lossless': True}, True),
    ],
)
def test_image_codec_roundtrip_in_fresh_process(tmp_path, codec_name, options, lossless):
    imagecodecs = pytest.importorskip('imagecodecs')
    codecs = pytest.importorskip('imagecodecs.numcodecs')
    if not getattr(imagecodecs, codec_name.upper()).available:
        pytest.skip(f'{codec_name} backend unavailable')
    compressor = getattr(codecs, codec_name)(**options)
    image = np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    expected = np.asarray(compressor.decode(compressor.encode(image))).reshape(image.shape)
    if lossless:
        np.testing.assert_array_equal(expected, image)
    frames = np.stack([image, image])
    dataset_path = tmp_path / 'images.zarr'
    buffer = ReplayBuffer.create_from_path(str(dataset_path), mode='w')
    buffer.add_episode(
        {'camera_rgb': frames}, chunks={'camera_rgb': (1, 32, 32, 3)},
        compressors=compressor,
    )
    np.save(tmp_path / 'expected.npy', np.stack([expected, expected]))

    _run_python('''
import sys
import numpy as np
from common.replay_buffer import ReplayBuffer
expected = np.load(sys.argv[2])
buffer = ReplayBuffer.create_from_path(sys.argv[1], mode='r')
np.testing.assert_array_equal(buffer['camera_rgb'][:], expected)
''', dataset_path, tmp_path / 'expected.npy')


@pytest.mark.parametrize(
    'compressor',
    [
        numcodecs.Blosc(cname='zstd', clevel=1),
        numcodecs.Blosc(cname='zstd', clevel=9),
        numcodecs.Blosc(cname='lz4', clevel=5),
        numcodecs.Blosc(cname='lz4hc', clevel=5),
        numcodecs.Zstd(level=3),
        numcodecs.Zstd(level=15),
        None,
    ],
)
def test_builtin_codec_disk_roundtrip(tmp_path, compressor):
    dataset_path = tmp_path / 'builtin.zarr'
    frames = np.random.default_rng(42).integers(0, 256, (2, 16, 16, 3), dtype=np.uint8)
    buffer = ReplayBuffer.create_from_path(str(dataset_path), mode='w')
    buffer.add_episode({'camera_rgb': frames}, compressors=compressor)

    loaded = ReplayBuffer.create_from_path(str(dataset_path), mode='r')

    np.testing.assert_array_equal(loaded['camera_rgb'][:], frames)


def test_registration_preserves_existing_codec():
    pytest.importorskip('imagecodecs.numcodecs')
    _run_python('''
from imagecodecs.numcodecs import Jpegxl
from numcodecs import register_codec
from numcodecs.registry import codec_registry
class ExistingJpegxl(Jpegxl):
    pass
register_codec(ExistingJpegxl)
from common.replay_buffer import ReplayBuffer
assert codec_registry['imagecodecs_jpegxl'] is ExistingJpegxl
''')


def test_builtin_codecs_work_without_imagecodecs():
    result = _run_python('''
import sys
sys.modules['imagecodecs'] = None
import numpy as np
from common.replay_buffer import ReplayBuffer
buffer = ReplayBuffer.create_empty_zarr()
values = np.arange(12).reshape(4, 3)
buffer.add_episode({'state': values})
np.testing.assert_array_equal(buffer['state'][:], values)
''')
    assert 'python -m pip install imagecodecs' in result.stderr
