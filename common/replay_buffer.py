from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property
from multiprocessing import Process, Lock
from zarr.storage import MemoryStore, LocalStore


def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0

def rechunk_recompress_array(group, name, 
        chunks=None, chunk_length=None,
        compressor=None, tmp_key='_temp'):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)
    
    if compressor is None:
        compressor = old_arr.compressor
    
    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # zarr 3.x: group.move() is not available, use method of reading data to memory, deleting old array, then creating new array
    data_np = old_arr[:]
    del group[name]
    # zarr 3.x: if data is passed, cannot pass shape and dtype at the same time (will be inferred from data)
    # Use create_array to create new array, supports zarr_format=2 and compressor
    arr = group.create_array(
        name=name,
        data=data_np,
        chunks=chunks,
        compressor=compressor
    )
    return arr

def get_optimal_chunks(shape, dtype, 
        target_chunk_bytes=2e6, 
        max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape)-1
    for i in range(len(shape)-1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[:i+1])
        if this_chunk_bytes <= target_chunk_bytes \
            and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """
    def __init__(self, 
            root: Union[zarr.Group, 
            Dict[str,dict]]):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])
        # zarr 3.x: Group does not have items() method, use keys() instead
        if isinstance(root['data'], zarr.Group):
            for key in root['data'].keys():
                value = root['data'][key]
                if value.shape[0] != root['meta']['episode_ends'][-1]:
                    raise ValueError(f"Episode ends {root['meta']['episode_ends'][-1]} does not match data length {value.shape[0]} for key {key}")
        else:
            for key, value in root['data'].items():
                if value.shape[0] != root['meta']['episode_ends'][-1]:
                    raise ValueError(f"Episode ends {root['meta']['episode_ends'][-1]} does not match data length {value.shape[0]} for key {key}")
        self.root = root
        self.lock = Lock()  # for multi-process writing 
    
    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = MemoryStore()
            root = zarr.group(store=storage, zarr_format=2)
        data = root.require_group('data', overwrite=False)
        meta = root.require_group('meta', overwrite=False)
        if 'episode_ends' not in meta:
            # Use create_array instead of zeros to support zarr_format=2 and compressor
            episode_ends = meta.create_array(name='episode_ends', shape=(0,), dtype=np.int64,
                fill_value=0, compressor=None, overwrite=False)
        return cls(root=root)
    
    @classmethod
    def create_empty_numpy(cls):
        root = {
            'data': dict(),
            'meta': {
                'episode_ends': np.zeros((0,), dtype=np.int64)
            }
        }
        return cls(root=root)
    
    @classmethod
    def create_from_group(cls, group, **kwargs):
        if 'data' not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode='r', **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        store_path = os.path.expanduser(zarr_path)
        fmt2_meta = os.path.exists(os.path.join(store_path, '.zgroup'))
        fmt3_meta = os.path.exists(os.path.join(store_path, 'zarr.json'))
        # If dataset does not exist or using writable mode, force zarr_format=2 to avoid writing format3
        needs_format2 = mode in ('w', 'w-', 'a', 'r+') or fmt2_meta
        if fmt3_meta and not fmt2_meta and needs_format2:
            raise RuntimeError(
                f"{zarr_path} already exists as Zarr format 3, cannot use compressor. "
                "Please delete the directory or use zarr >=3 codec pipeline."
            )
        # zarr 3.x: directly use path string, will automatically use LocalStore
        open_kwargs = dict(store=store_path, mode=mode)
        if needs_format2:
            open_kwargs['zarr_format'] = 2
        group = zarr.open_group(**open_kwargs)
        return cls.create_from_group(group, **kwargs)
    
    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(cls, src_store, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Load to memory.
        """
        src_root = zarr.group(src_store)
        root = None
        if store is None:
            # numpy backend
            meta = dict()
            # zarr 3.x: Group does not have items() method, use keys() instead
            if isinstance(src_root['meta'], zarr.Group):
                meta_keys = list(src_root['meta'].keys())
            else:
                meta_keys = list(src_root['meta'].keys())
            for key in meta_keys:
                value = src_root['meta'][key]
                if isinstance(value, zarr.Group):
                    continue
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                # zarr 3.x: Group does not have items() method, use keys() instead
                if isinstance(src_root['data'], zarr.Group):
                    keys = list(src_root['data'].keys())
                else:
                    keys = list(src_root['data'].keys())
            data = dict()
            for key in keys:
                arr = src_root['data'][key]
                data[key] = arr[:]

            root = {
                'meta': meta,
                'data': data
            }
        else:
            root = zarr.group(store=store, zarr_format=2)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(source=src_store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
            data_group = root.create_group('data', overwrite=True)
            if keys is None:
                # zarr 3.x: Group does not have items() method, use keys() instead
                if isinstance(src_root['data'], zarr.Group):
                    keys = list(src_root['data'].keys())
                else:
                    keys = list(src_root['data'].keys())
            for key in keys:
                value = src_root['data'][key]
                cks = cls._resolve_array_chunks(
                    chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store, dest=store,
                        source_path=this_path, dest_path=this_path,
                        if_exists=if_exists
                    )
                else:
                    # zarr 3.x: zarr.copy() does not support compressor, use method of reading data to memory then creating new array
                    data_np = value[:]
                    if if_exists == 'replace' and key in data_group:
                        del data_group[key]
                    # zarr 3.x: if data is passed, cannot pass shape and dtype at the same time
                    _ = data_group.create_array(
                        name=key,
                        data=data_np,
                        chunks=cks,
                        compressor=cpr
                    )
        buffer = cls(root=root)
        return buffer
    
    @classmethod
    def copy_from_path(cls, zarr_path, backend=None, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        if backend == 'numpy':
            print('backend argument is deprecated!')
            store = None
        store_path = os.path.expanduser(zarr_path)
        # zarr 3.x: directly use path string, will automatically use LocalStore
        open_kwargs = dict(store=store_path, mode='r')
        if os.path.exists(os.path.join(store_path, '.zgroup')):
            open_kwargs['zarr_format'] = 2
        group = zarr.open_group(**open_kwargs)
        return cls.copy_from_store(src_store=group.store, store=store, 
            keys=keys, chunks=chunks, compressors=compressors, 
            if_exists=if_exists, **kwargs)

    # ============= save methods ===============
    def save_to_store(self, store, 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(),
            if_exists='replace', 
            **kwargs):
        
        root = zarr.group(store=store, zarr_format=2)
        if self.backend == 'zarr':
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
        else:
            meta_group = root.create_group('meta', overwrite=True)
            # save meta, no chunking
            for key, value in self.root['meta'].items():
                # zarr 3.x: if data is passed, cannot pass shape and dtype at the same time
                _ = meta_group.create_array(
                    name=key,
                    data=value, 
                    chunks=value.shape)
        
        # save data, chunk
        data_group = root.create_group('data', overwrite=True)
        # zarr 3.x: Group does not have items() method, use keys() instead
        if isinstance(self.root['data'], zarr.Group):
            data_keys = self.root['data'].keys()
        else:
            data_keys = self.root['data'].keys()
        for key in data_keys:
            value = self.root['data'][key]
            cks = self._resolve_array_chunks(
                chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store, dest=store,
                        source_path=this_path, dest_path=this_path, if_exists=if_exists)
                else:
                    # zarr 3.x: zarr.copy() does not support compressor, use method of reading data to memory then creating new array
                    data_np = value[:]
                    if if_exists == 'replace' and key in data_group:
                        del data_group[key]
                    # zarr 3.x: if data is passed, cannot pass shape and dtype at the same time
                    _ = data_group.create_array(
                        name=key,
                        data=data_np,
                        chunks=cks,
                        compressor=cpr
                    )
            else:
                # numpy
                # zarr 3.x: if data is passed, cannot pass shape and dtype at the same time
                _ = data_group.create_array(
                    name=key,
                    data=value,
                    chunks=cks,
                    compressor=cpr
                )
        return store

    def save_to_path(self, zarr_path,             
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(), 
            if_exists='replace', 
            **kwargs):
        # zarr 3.x: directly use path string, will automatically use LocalStore
        store = LocalStore(os.path.expanduser(zarr_path))
        return self.save_to_store(store, chunks=chunks, 
            compressors=compressors, if_exists=if_exists, **kwargs)

    @staticmethod
    def resolve_compressor(compressor='default'):
        if compressor == 'default':
            compressor = numcodecs.Blosc(cname='lz4', clevel=5, 
                shuffle=numcodecs.Blosc.NOSHUFFLE)
        elif compressor == 'disk':
            compressor = numcodecs.Blosc('zstd', clevel=5, 
                shuffle=numcodecs.Blosc.BITSHUFFLE)
        return compressor

    @classmethod
    def _resolve_array_compressor(cls, 
            compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        # allows compressor to be explicitly set to None
        cpr = 'nil'
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == 'nil':
            cpr = cls.resolve_compressor('default')
        return cpr
    
    @classmethod
    def _resolve_array_chunks(cls,
            chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks
    
    # ============= properties =================
    @cached_property
    def data(self):
        return self.root['data']
    
    @cached_property
    def meta(self):
        return self.root['meta']

    def update_meta(self, data):
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == 'zarr':
            for key, value in np_data.items():
                # zarr 3.x: if data is passed, cannot pass shape and dtype at the same time
                _ = meta_group.create_array(
                    name=key,
                    data=value, 
                    chunks=value.shape,
                    overwrite=True)
        else:
            meta_group.update(np_data)
        
        return meta_group
    
    @property
    def episode_ends(self):
        return self.meta['episode_ends']
    
    def get_episode_idxs(self):
        import numba
        # zarr 3.x: Array does not support len(), use shape[0]
        episode_ends_array = self.episode_ends[:]  # Convert to numpy array
        numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result
        return _get_episode_idxs(episode_ends_array)
        
    
    @property
    def backend(self):
        backend = 'numpy'
        if isinstance(self.root, zarr.Group):
            backend = 'zarr'
        return backend
    
    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == 'zarr':
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        # zarr 3.x: Group does not have items() method, use keys() instead
        if isinstance(self.data, zarr.Group):
            return ((key, self.data[key]) for key in self.data.keys())
        else:
            return self.data.items()
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        # zarr 3.x: Array does not support len(), use shape[0]
        if self.episode_ends.shape[0] == 0:
            return 0
        return self.episode_ends[-1]
    
    @property
    def n_episodes(self):
        # zarr 3.x: Array does not support len(), use shape[0]
        return self.episode_ends.shape[0]

    @property
    def chunk_size(self):
        if self.backend == 'zarr':
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def _add_episode(self, 
            data: Dict[str, np.ndarray], 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict()):
        
        assert(len(data) > 0)
        is_zarr = (self.backend == 'zarr')

        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert(len(value.shape) >= 1)
            if episode_length is None:
                episode_length = len(value)
            else:
                assert(episode_length == len(value))
        new_len = int(curr_len + episode_length)

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            new_shape = tuple(int(x) for x in new_shape)
            # create array
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value)
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value)
                    # zarr 3.x: use create_array instead of zeros to support zarr_format=2 and compressor
                    arr = self.data.create_array(name=key, 
                        shape=new_shape, 
                        dtype=value.dtype,
                        fill_value=0,
                        chunks=cks,
                        compressor=cpr)
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert(value.shape[1:] == arr.shape[1:])
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy data
            arr[-value.shape[0]:] = value
        
        # append to episode ends
        episode_ends = self.episode_ends
        if is_zarr:
            # zarr 3.x: ensure resize parameter is Python int
            new_size = int(episode_ends.shape[0] + 1)
            episode_ends.resize(new_size)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

        # rechunk
        if is_zarr:
            if episode_ends.chunks[0] < episode_ends.shape[0]:
                rechunk_recompress_array(self.meta, 'episode_ends', 
                    chunk_length=int(episode_ends.shape[0] * 1.5))


    def add_episode(self, 
            data: Dict[str, np.ndarray], 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict()):

        with self.lock:
            self._add_episode(data, chunks=chunks, compressors=compressors)

    
    def drop_episode(self):
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        assert(len(episode_ends) > 0)
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        # zarr 3.x: Group does not have items() method, use keys() instead
        if isinstance(self.data, zarr.Group):
            for key in self.data.keys():
                value = self.data[key]
                new_shape = (start_idx,) + value.shape[1:]
                new_shape = tuple(int(x) for x in new_shape)
                if is_zarr:
                    value.resize(new_shape)
                else:
                    value.resize(new_shape, refcheck=False)
        else:
            for key, value in self.data.items():
                new_shape = (start_idx,) + value.shape[1:]
                new_shape = tuple(int(x) for x in new_shape)  # zarr 3.x: ensure all elements are Python int
                if is_zarr:
                    value.resize(new_shape)
                else:
                    value.resize(new_shape, refcheck=False)
        if is_zarr:
            # zarr 3.x: Array does not support len(), use shape[0], and ensure it is Python int
            new_size = int(self.episode_ends.shape[0] - 1)
            self.episode_ends.resize(new_size)
        else:
            self.episode_ends.resize(len(episode_ends)-1, refcheck=False)
    
    def pop_episode(self):
        assert(self.n_episodes > 0)
        episode = self.get_episode(self.n_episodes-1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        # zarr 3.x: Array does not support len(), use shape[0]
        idx = list(range(self.episode_ends.shape[0]))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result
    
    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        # zarr 3.x: Group does not have items() method, use keys() instead
        if isinstance(self.data, zarr.Group):
            for key in self.data.keys():
                value = self.data[key]
                x = value[_slice]
                if copy and isinstance(value, np.ndarray):
                    x = x.copy()
                result[key] = x
        else:
            for key, value in self.data.items():
                x = value[_slice]
                if copy and isinstance(value, np.ndarray):
                    x = x.copy()
                result[key] = x
        return result
    
    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == 'zarr'
        chunks = dict()
        # zarr 3.x: Group does not have items() method, use keys() instead
        for key in self.data.keys():
            value = self.data[key]
            chunks[key] = value.chunks
        return chunks
    
    def set_chunks(self, chunks: dict):
        assert self.backend == 'zarr'
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == 'zarr'
        compressors = dict()
        # zarr 3.x: Group does not have items() method, use keys() instead
        for key in self.data.keys():
            value = self.data[key]
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == 'zarr'
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)