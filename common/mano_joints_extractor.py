"""
MANO Joints 3D Position Extractor

This module provides tools to extract 21 joint 3D positions from Phase 2b MANO HDF5 data.

Usage:
    from tools.mano_joints_extractor import MANOJointsExtractor

    extractor = MANOJointsExtractor(device='cuda:0')
    result = extractor.extract_from_file('episode_xxx_mano.hdf5', hand='right')
    joints_3d = result['joints_3d']  # (T, 21, 3)
"""

import os
import sys
import json
import glob
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

try:
    from manotorch.manolayer import ManoLayer
except ImportError as exc:
    raise ImportError(
        "manotorch is not importable. Install it in the environment or set "
        "PAXINI_REPO_ROOT to a checkout containing "
        "'px_retargeting/assets/packages/manotorch'."
    ) from exc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MANO joint names in SNAP order (after reordering in manolayer.py line 240)
MANO_JOINT_NAMES = [
    'wrist',        #  0
    'thumb_0',      #  1  (CMC)
    'thumb_1',      #  2  (MCP)
    'thumb_2',      #  3  (IP)
    'thumb_tip',    #  4  (tip)
    'index_0',      #  5  (MCP)
    'index_1',      #  6  (PIP)
    'index_2',      #  7  (DIP)
    'index_tip',    #  8  (tip)
    'middle_0',     #  9
    'middle_1',     # 10
    'middle_2',     # 11
    'middle_tip',   # 12
    'ring_0',       # 13
    'ring_1',       # 14
    'ring_2',       # 15
    'ring_tip',     # 16
    'pinky_0',      # 17
    'pinky_1',      # 18
    'pinky_2',      # 19
    'pinky_tip'     # 20
]


class MANOJointsExtractor:
    """Extract 3D joint positions from MANO Phase 2b data.

    This class provides functionality to:
    - Load MANO configuration and shape parameters (betas)
    - Process Phase 2b HDF5 files containing 48-dim MANO pose parameters
    - Compute 21 joint 3D positions using manotorch
    - Optionally transform joints to world coordinates
    - Support batch processing of multiple files

    Attributes:
        device (str): PyTorch device ('cuda:0', 'cpu', etc.)
        batch_size (int): Batch size for GPU inference
        mano_layer_right (ManoLayer): Right hand MANO model
        mano_layer_left (ManoLayer): Left hand MANO model
        config_right (dict): Right hand configuration
        config_left (dict): Left hand configuration
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        betas: Optional[np.ndarray] = None,
        device: str = 'cuda:0',
        batch_size: int = 32
    ):
        """Initialize the MANO joints extractor.

        Args:
            config_path: Path to MANO config file. If None, auto-detects from repository.
            betas: Custom 10-dim shape parameters. If None, loads from config file.
            device: PyTorch device for computation
            batch_size: Batch size for GPU inference
        """
        self.device = device
        self.batch_size = batch_size

        if REPO_ROOT is None:
            raise RuntimeError(
                "PAXINI_REPO_ROOT is not set and a local 'third_party/paxini' checkout was not found. "
                "This extractor needs Paxini MANO assets to load configs and meshes."
            )

        # Paths
        self.config_dir = REPO_ROOT / "px_retargeting" / "assets" / "config" / "hand_model"
        self.mano_assets_root = (
            REPO_ROOT / "px_retargeting" / "assets" / "urdfs" / "mano_hand_description" / "mano_v1_2"
        )

        # Load configurations
        self.config_right = self._load_config('right')
        self.config_left = self._load_config('left')

        # Override betas if provided
        if betas is not None:
            if betas.shape != (10,):
                raise ValueError(f"betas must be shape (10,), got {betas.shape}")
            self.config_right['betas'] = betas.tolist()
            self.config_left['betas'] = betas.tolist()

        # Initialize MANO layers
        logger.info(f"Initializing MANO layers on device: {device}")
        self.mano_layer_right = self._init_mano_layer('right')
        self.mano_layer_left = self._init_mano_layer('left')

        logger.info("MANOJointsExtractor initialized successfully")

    def _load_config(self, hand: str) -> dict:
        """Load MANO configuration file.

        Args:
            hand: 'right' or 'left'

        Returns:
            Configuration dictionary
        """
        config_file = self.config_dir / f'mano_{hand[0]}h.json'

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, 'r') as f:
            config = json.load(f)

        logger.info(f"Loaded {hand} hand config from: {config_file}")
        return config

    def _init_mano_layer(self, hand: str) -> ManoLayer:
        """Initialize ManoLayer for specified hand.

        Args:
            hand: 'right' or 'left'

        Returns:
            Initialized ManoLayer
        """
        config = self.config_right if hand == 'right' else self.config_left
        betas = torch.tensor(config['betas'], dtype=torch.float32).unsqueeze(0)

        # Initialize MANO layer
        mano_layer = ManoLayer(
            rot_mode='axisang',
            use_pca=False,
            side=hand,
            center_idx=None,
            mano_assets_root=str(self.mano_assets_root),
            flat_hand_mean=True
        )

        # Move to device
        mano_layer = mano_layer.to(self.device)
        mano_layer.eval()

        # Store betas on device
        mano_layer.default_betas = betas.to(self.device)

        return mano_layer

    def extract_from_file(
        self,
        hdf5_path: str,
        hand: str = 'right',
        apply_global_transform: bool = True,
        output_format: str = 'dict'
    ) -> Dict[str, np.ndarray]:
        """Extract joint 3D positions from a single HDF5 file.

        Args:
            hdf5_path: Path to Phase 2b HDF5 file
            hand: 'right' or 'left'
            apply_global_transform: If True, transform to world coordinates
            output_format: 'dict', 'npz', or 'hdf5'

        Returns:
            Dictionary containing:
                - joints_3d: (T, 21, 3) array of joint positions
                - joint_names: List of 21 joint names
                - betas: (10,) shape parameters used
                - metadata: Additional information
        """
        # Validate inputs
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        if hand not in ['right', 'left']:
            raise ValueError(f"hand must be 'right' or 'left', got {hand}")

        logger.info(f"Processing file: {hdf5_path}")
        logger.info(f"Hand: {hand}, Global transform: {apply_global_transform}")

        # Read data from HDF5
        joint_angles, hand_pose = self._read_hdf5(hdf5_path, hand)
        T = joint_angles.shape[0]
        logger.info(f"Read {T} frames from HDF5")

        # Get MANO layer and config
        mano_layer = self.mano_layer_right if hand == 'right' else self.mano_layer_left
        config = self.config_right if hand == 'right' else self.config_left
        betas = np.array(config['betas'])

        # Compute joints in local MANO frame
        joints_local = self._compute_mano_joints(joint_angles, mano_layer)
        logger.info(f"Computed local joints: {joints_local.shape}")

        # Apply global transformation if requested
        if apply_global_transform:
            joints_3d = self._apply_global_transform(
                joints_local, hand_pose, config
            )
            coordinate_frame = 'world'
        else:
            joints_3d = joints_local
            coordinate_frame = 'local'

        logger.info(f"Final joints shape: {joints_3d.shape}")

        # Prepare result
        result = {
            'joints_3d': joints_3d,
            'joint_names': MANO_JOINT_NAMES,
            'betas': betas,
            'metadata': {
                'num_frames': T,
                'hand': hand,
                'coordinate_frame': coordinate_frame,
                'source_file': os.path.basename(hdf5_path)
            }
        }

        # Save to file if requested
        if output_format == 'npz':
            output_path = hdf5_path.replace('.hdf5', '_joints.npz')
            self._save_npz(result, output_path)
            logger.info(f"Saved to: {output_path}")
        elif output_format == 'hdf5':
            output_path = hdf5_path.replace('.hdf5', '_with_joints.hdf5')
            self._save_hdf5(result, hdf5_path, output_path, hand)
            logger.info(f"Saved to: {output_path}")

        return result

    def _read_hdf5(self, hdf5_path: str, hand: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read joint angles and hand pose from HDF5.

        Args:
            hdf5_path: Path to HDF5 file
            hand: 'right' or 'left'

        Returns:
            Tuple of (joint_angles, hand_pose)
                - joint_angles: (T, 48) MANO pose parameters
                - hand_pose: (T, 7) wrist 6D pose [px, py, pz, qw, qx, qy, qz]
        """
        hand_key = 'righthand' if hand == 'right' else 'lefthand'

        with h5py.File(hdf5_path, 'r') as f:
            # Check required datasets exist
            joints_key = f'dataset/observation/{hand_key}/joints/data'
            handpose_key = f'dataset/observation/{hand_key}/handpose/data'

            if joints_key not in f:
                raise ValueError(f"Missing required dataset: {joints_key}")
            if handpose_key not in f:
                raise ValueError(f"Missing required dataset: {handpose_key}")

            # Read data
            joint_angles = f[joints_key][:]  # (T, 48)
            hand_pose = f[handpose_key][:]   # (T, 7)

        # Validate shapes
        if joint_angles.ndim != 2 or joint_angles.shape[1] != 48:
            raise ValueError(f"Expected joint_angles shape (T, 48), got {joint_angles.shape}")
        if hand_pose.ndim != 2 or hand_pose.shape[1] != 7:
            raise ValueError(f"Expected hand_pose shape (T, 7), got {hand_pose.shape}")

        return joint_angles, hand_pose

    def _compute_mano_joints(
        self,
        joint_angles: np.ndarray,
        mano_layer: ManoLayer
    ) -> np.ndarray:
        """Compute joint 3D positions using MANO forward pass.

        Args:
            joint_angles: (T, 48) MANO pose parameters
            mano_layer: Initialized ManoLayer

        Returns:
            joints: (T, 21, 3) joint 3D positions in local frame
        """
        T = joint_angles.shape[0]
        all_joints = []

        # Process in batches
        for batch_start in range(0, T, self.batch_size):
            batch_end = min(batch_start + self.batch_size, T)
            batch_size_actual = batch_end - batch_start

            # Prepare batch data
            pose_batch = torch.from_numpy(
                joint_angles[batch_start:batch_end]
            ).float().to(self.device)

            # Repeat betas for batch
            betas_batch = mano_layer.default_betas.repeat(batch_size_actual, 1)

            # Forward pass
            with torch.no_grad():
                try:
                    mano_output = mano_layer(pose_batch, betas_batch)
                    joints_batch = mano_output.joints  # (B, 21, 3)
                    all_joints.append(joints_batch.cpu().numpy())
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"GPU OOM at batch {batch_start}-{batch_end}")
                        logger.error(f"Try reducing batch_size (current: {self.batch_size})")
                        # Clear cache and retry with smaller batch
                        torch.cuda.empty_cache()
                        raise
                    else:
                        raise

        # Concatenate all batches
        joints = np.concatenate(all_joints, axis=0)
        return joints

    def _apply_global_transform(
        self,
        joints_local: np.ndarray,
        hand_pose: np.ndarray,
        config: dict
    ) -> np.ndarray:
        """Transform joints from local MANO frame to world frame.

        Args:
            joints_local: (T, 21, 3) joints in local frame
            hand_pose: (T, 7) wrist pose [px, py, pz, qw, qx, qy, qz]
            config: MANO configuration dict

        Returns:
            joints_world: (T, 21, 3) joints in world frame
        """
        T = joints_local.shape[0]
        joints_world = np.zeros_like(joints_local)

        # Get base transform from config
        base_tf = np.array(config['base_tf_rel_wrist'])  # (4, 4)

        for t in range(T):
            # Extract quaternion and position
            quat_xyzw = hand_pose[t, [4,5,6,3]]  # [qx, qy, qz, qw]
            position = hand_pose[t, :3]   # [px, py, pz]

            # Quaternion to rotation matrix (scipy uses xyzw order)
            rot_mat = R.from_quat(quat_xyzw).as_matrix()

            # Build wrist transform matrix
            wrist_tf = np.eye(4)
            wrist_tf[:3, :3] = rot_mat
            wrist_tf[:3, 3] = position

            # Combine transforms: world = wrist_tf @ base_tf @ local
            full_transform = wrist_tf @ base_tf

            # Apply to all joints
            joints_homogeneous = np.hstack([
                joints_local[t],
                np.ones((21, 1))
            ])  # (21, 4)

            joints_world[t] = (wrist_tf @ joints_homogeneous.T).T[:, :3]

        return joints_world

    def _save_npz(self, result: dict, output_path: str):
        """Save result to NPZ format.

        Args:
            result: Result dictionary
            output_path: Output file path
        """
        np.savez_compressed(
            output_path,
            joints_3d=result['joints_3d'],
            joint_names=result['joint_names'],
            betas=result['betas'],
            num_frames=result['metadata']['num_frames'],
            hand=result['metadata']['hand'],
            coordinate_frame=result['metadata']['coordinate_frame'],
            source_file=result['metadata']['source_file']
        )

    def _save_hdf5(
        self,
        result: dict,
        source_path: str,
        output_path: str,
        hand: str
    ):
        """Save result by extending original HDF5 file.

        Args:
            result: Result dictionary
            source_path: Original HDF5 file
            output_path: Output file path
            hand: 'right' or 'left'
        """
        # Copy original file
        import shutil
        shutil.copy2(source_path, output_path)

        # Add joints_3d dataset
        hand_key = 'righthand' if hand == 'right' else 'lefthand'

        with h5py.File(output_path, 'a') as f:
            # Create joints_3d group
            group_path = f'dataset/observation/{hand_key}/joints_3d'
            if group_path in f:
                del f[group_path]  # Remove if exists

            group = f.create_group(group_path)
            group.create_dataset('data', data=result['joints_3d'], compression='gzip')

            # Add metadata as attributes
            group.attrs['betas'] = result['betas']
            group.attrs['coordinate_frame'] = result['metadata']['coordinate_frame']
            group.attrs['joint_names'] = ','.join(result['joint_names'])

    def extract_batch(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str = 'npz',
        pattern: str = '*_mano.hdf5',
        hand: str = 'right',
        apply_global_transform: bool = True
    ) -> List[str]:
        """Batch process multiple HDF5 files.

        Args:
            input_dir: Directory containing Phase 2b HDF5 files
            output_dir: Output directory
            output_format: 'npz' or 'hdf5'
            pattern: File matching pattern
            hand: 'right' or 'left'
            apply_global_transform: Whether to apply global transform

        Returns:
            List of successfully processed file paths
        """
        # Find all matching files
        search_pattern = os.path.join(input_dir, pattern)
        hdf5_files = sorted(glob.glob(search_pattern))

        if not hdf5_files:
            logger.warning(f"No files found matching: {search_pattern}")
            return []

        logger.info(f"Found {len(hdf5_files)} files to process")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process each file
        successful = []
        failed = []

        for i, hdf5_path in enumerate(hdf5_files, 1):
            try:
                logger.info(f"\n[{i}/{len(hdf5_files)}] Processing: {os.path.basename(hdf5_path)}")

                # Extract joints
                result = self.extract_from_file(
                    hdf5_path,
                    hand=hand,
                    apply_global_transform=apply_global_transform,
                    output_format='dict'  # Get result first
                )

                # Save to output directory
                basename = os.path.basename(hdf5_path)
                if output_format == 'npz':
                    output_path = os.path.join(
                        output_dir,
                        basename.replace('.hdf5', '_joints.npz')
                    )
                    self._save_npz(result, output_path)
                elif output_format == 'hdf5':
                    output_path = os.path.join(
                        output_dir,
                        basename.replace('.hdf5', '_with_joints.hdf5')
                    )
                    self._save_hdf5(result, hdf5_path, output_path, hand)
                else:
                    raise ValueError(f"Unsupported output_format: {output_format}")

                successful.append(output_path)
                logger.info(f"✓ Saved to: {output_path}")

            except Exception as e:
                logger.error(f"✗ Failed to process {hdf5_path}: {e}")
                failed.append(hdf5_path)

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch processing complete!")
        logger.info(f"Successful: {len(successful)}/{len(hdf5_files)}")
        if failed:
            logger.info(f"Failed: {len(failed)}")
            for f in failed:
                logger.info(f"  - {f}")
        logger.info(f"{'='*60}")

        return successful


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract MANO joint 3D positions from Phase 2b HDF5 data'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input HDF5 file or directory'
    )
    parser.add_argument(
        '--output',
        help='Output file or directory (default: same as input with suffix)'
    )
    parser.add_argument(
        '--hand',
        choices=['right', 'left'],
        default='right',
        help='Hand to process (default: right)'
    )
    parser.add_argument(
        '--format',
        choices=['dict', 'npz', 'hdf5'],
        default='npz',
        help='Output format (default: npz)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for GPU inference (default: 32)'
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='PyTorch device (default: cuda:0)'
    )
    parser.add_argument(
        '--no-global-transform',
        action='store_true',
        help='Do not apply global transformation (keep in local MANO frame)'
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = MANOJointsExtractor(
        device=args.device,
        batch_size=args.batch_size
    )

    # Process file or directory
    if os.path.isfile(args.input):
        # Single file
        result = extractor.extract_from_file(
            args.input,
            hand=args.hand,
            apply_global_transform=not args.no_global_transform,
            output_format=args.format
        )
        print(f"\nExtracted {result['joints_3d'].shape[0]} frames")
        print(f"Output saved")

    elif os.path.isdir(args.input):
        # Batch processing
        output_dir = args.output or os.path.join(args.input, 'joints_output')

        extractor.extract_batch(
            input_dir=args.input,
            output_dir=output_dir,
            output_format=args.format,
            hand=args.hand,
            apply_global_transform=not args.no_global_transform
        )
    else:
        print(f"Error: {args.input} is neither a file nor directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
