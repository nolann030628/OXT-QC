import numpy as np
import numcodecs
import pytest

from common.replay_buffer import ReplayBuffer


def test_resolve_compressor_defaults_to_disk():
    compressor = ReplayBuffer.resolve_compressor()

    assert compressor.cname == 'zstd'
    assert compressor.clevel == 5
    assert compressor.shuffle == numcodecs.Blosc.BITSHUFFLE


@pytest.mark.parametrize(
    'compressors, expected_cname, expected_shuffle',
    [
        ({}, 'zstd', numcodecs.Blosc.BITSHUFFLE),
        ('disk', 'zstd', numcodecs.Blosc.BITSHUFFLE),
        ('default', 'lz4', numcodecs.Blosc.NOSHUFFLE),
        (None, None, None),
    ],
)
def test_add_episode_compressor_selection(
    compressors, expected_cname, expected_shuffle
):
    buffer = ReplayBuffer.create_empty_zarr()
    episode = {'state': np.arange(12, dtype=np.float32).reshape(4, 3)}

    buffer.add_episode(episode, compressors=compressors)

    compressor = buffer.data['state'].compressor
    if expected_cname is None:
        assert compressor is None
    else:
        assert compressor.cname == expected_cname
        assert compressor.shuffle == expected_shuffle
    np.testing.assert_array_equal(buffer.data['state'][:], episode['state'])


def test_add_episode_preserves_existing_compressor():
    buffer = ReplayBuffer.create_empty_zarr()
    episode = {'state': np.arange(12, dtype=np.float32).reshape(4, 3)}
    buffer.add_episode(episode, compressors='default')

    buffer.add_episode(episode)

    assert buffer.data['state'].compressor.cname == 'lz4'
    np.testing.assert_array_equal(
        buffer.data['state'][:], np.concatenate([episode['state']] * 2)
    )
    compressor = ReplayBuffer._resolve_array_compressor(
        {}, 'state', buffer.data['state']
    )
    assert compressor.cname == 'lz4'


def test_partial_compressor_mapping_defaults_other_fields_to_disk():
    buffer = ReplayBuffer.create_empty_zarr()
    values = np.arange(12, dtype=np.float32).reshape(4, 3)

    buffer.add_episode(
        {'state': values, 'other_state': values},
        compressors={'state': 'default'},
    )

    assert buffer.data['state'].compressor.cname == 'lz4'
    assert buffer.data['other_state'].compressor.cname == 'zstd'
