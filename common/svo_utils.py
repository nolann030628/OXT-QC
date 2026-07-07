import pickle
import numpy as np 
import cv2 
import pyzed.sl as sl
from typing import Sequence, Tuple, Dict, Optional, Union, Generator
import multiprocessing
import os
import pathlib
import yaml
import click
import shutil
from tqdm import tqdm
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from common.cv2_utils import get_image_transform_resize_crop
from common.timestamp_accumulator import get_accumulate_timestamp_idxs


class SVOReader:
    def __init__(self, filepath, serial_number):
        # Save Parameters #
        self.serial_number = serial_number
        self._index = 0

        # Initialize Readers #
        self._sbs_img = sl.Mat()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._left_depth = sl.Mat()
        self._right_depth = sl.Mat()
        self._left_pointcloud = sl.Mat()
        self._right_pointcloud = sl.Mat()

        # Set SVO path for playback
        init_parameters = sl.InitParameters()
        init_parameters.camera_image_flip = sl.FLIP_MODE.OFF
        init_parameters.depth_mode = sl.DEPTH_MODE.ULTRA 
        init_parameters.coordinate_units = sl.UNIT.METER
        init_parameters.set_from_svo_file(filepath)
        init_parameters.enable_right_side_measure = True

        # Open the ZED
        self._cam = sl.Camera()
        status = self._cam.open(init_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Zed Error: " + repr(status))

    def set_reading_parameters(
        self,
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        # Save Parameters #
        self.image = image
        self.depth = depth
        self.pointcloud = pointcloud
        self.concatenate_images = concatenate_images
        if resize_func is not None:
            self.resize_func = cv2.resize
        else:
            self.resize_func = None

        if self.resize_func is None:
            self.zed_resolution = sl.Resolution(*resolution)
            self.resizer_resolution = (0, 0)
        else:
            self.zed_resolution = sl.Resolution(0, 0)
            self.resizer_resolution = resolution

        self.skip_reading = not any([image, depth, pointcloud])
        if self.skip_reading:
            return
    
    def get_camera_information(self):
        cam_param = self._cam.get_camera_information().camera_configuration.calibration_parameters
        stereo_trans = cam_param.stereo_transform.get_translation().get()
        stereo_orn = cam_param.stereo_transform.get_orientation().get()
        stereo_transform = np.eye(4)
        stereo_transform[:3, -1] = stereo_trans 
        stereo_transform[:3, :3] = R.from_quat(stereo_orn).as_matrix()
        left_cam_param = cam_param.left_cam 
        right_cam_param = cam_param.right_cam 
        left_intr = np.array([[left_cam_param.fx, 0., left_cam_param.cx], [0., left_cam_param.fy, left_cam_param.cy], [0., 0., 1.]])
        right_intr = np.array([[right_cam_param.fx, 0., right_cam_param.cx], [0., right_cam_param.fy, right_cam_param.cy], [0., 0., 1.]])
        cam_info = dict()
        cam_info['stereo_transform'] = stereo_transform
        cam_info['left_intrinsic'] = left_intr 
        cam_info['right_intrinsic'] = right_intr
        return cam_info

    def get_frame_resolution(self):
        camera_info = self._cam.get_camera_information().camera_configuration
        width = camera_info.resolution.width
        height = camera_info.resolution.height
        return (width, height)

    def get_frame_count(self):
        if self.skip_reading:
            return 0
        return self._cam.get_svo_number_of_frames()

    def set_frame_index(self, index):
        if self.skip_reading:
            return

        if index < self._index:
            self._cam.set_svo_position(index)
            self._index = index

        while self._index < index:
            self.read_camera(ignore_data=True)

    def _process_frame(self, frame):
        frame = deepcopy(frame.get_data())
        if self.resizer_resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resizer_resolution)

    def read_camera(self, ignore_data=False, correct_timestamp=None, return_timestamp=False):
        # Skip if Read Unnecesary #
        if self.skip_reading:
            return {}

        # Read Camera #
        self._index += 1
        err = self._cam.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            return None
        if ignore_data:
            return None

        # Check Image Timestamp #
        received_time = self._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        timestamp_error = (correct_timestamp is not None) and (correct_timestamp != received_time)

        if timestamp_error:
            print("Timestamps did not match...")
            return None

        # Return Data #
        data_dict = {}

        if self.image:
            if self.concatenate_images:
                self._cam.retrieve_image(self._sbs_img, sl.VIEW.SIDE_BY_SIDE, resolution=self.zed_resolution)
                data_dict["image"] = {self.serial_number: self._process_frame(self._sbs_img)}
            else:
                self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT, resolution=self.zed_resolution)
                self._cam.retrieve_image(self._right_img, sl.VIEW.RIGHT, resolution=self.zed_resolution)
                data_dict["image"] = {
                    self.serial_number + "_left": self._process_frame(self._left_img),
                    self.serial_number + "_right": self._process_frame(self._right_img),
                }
        if self.depth:
        	self._cam.retrieve_measure(self._left_depth, sl.MEASURE.DEPTH, resolution=self.zed_resolution)
        	self._cam.retrieve_measure(self._right_depth, sl.MEASURE.DEPTH_RIGHT, resolution=self.zed_resolution)
        	data_dict['depth'] = {
        		self.serial_number + '_left': self._left_depth.get_data().copy(),
        		self.serial_number + '_right': self._right_depth.get_data().copy()}
        if self.pointcloud:
            self._cam.retrieve_measure(self._left_pointcloud, sl.MEASURE.XYZRGBA, resolution=sl.Resolution(*self.resizer_resolution))
        #  self._cam.retrieve_measure(self._right_pointcloud, sl.MEASURE.XYZRGBA_RIGHT, resolution=self.resolution)
        data_dict['pointcloud'] = {
        		self.serial_number + '_left': self._left_pointcloud.get_data().copy(),
        		self.serial_number + '_right': self._left_pointcloud.get_data().copy()}
        # import pdb; pdb.set_trace()
        if return_timestamp:
            return data_dict, received_time
        return data_dict

    def disable_camera(self):
        if hasattr(self, "_cam"):
            self._cam.close()




def parse_svo_to_episode(svo_filepath, serial_id, 
                         episode, 
                         mode,
                         dt,
                         camera_name,
                         out_resolutions_resize: Union[None, tuple]=None,      # (width, height)
                         out_resolutions_crop: Union[None, tuple]=None,  
                         out_resolutions_crop_random_ratio: Union[None, tuple]=None,
                         out_resolutions_image_final: Union[None, tuple]=None, # (width, height)
                         ):
    svo_camera = SVOReader(svo_filepath, serial_id)
    svo_stereo, svo_depth, svo_pointcloud = False, False, False
    if mode in ['d', 'a']:
        svo_depth = True
    if mode in ['s', 'a']:
        svo_stereo = True
    if mode in ['p', 'a']:
        svo_pointcloud = True
    svo_camera.set_reading_parameters(image=True, depth=svo_depth, pointcloud=svo_pointcloud, concatenate_images=False)
    frame_count = svo_camera.get_frame_count()
    width, height = svo_camera.get_frame_resolution()

    obs_dict = dict()
    episode_length = episode['timestamps'].shape[0]
    if mode != 'o':
        if out_resolutions_crop_random_ratio is None:
            pass
        elif out_resolutions_crop_random_ratio[0] == 1.0 and out_resolutions_crop_random_ratio[1] == 1.0:
            pass
        else:
            raise ValueError(f"out_resolutions_crop_random_ratio must be None or (1.0, 1.0) for mode != 'o'. Current mode is {mode}, but got {out_resolutions_crop_random_ratio}.")
    episode[f'{camera_name}_real_timestamp'] = np.zeros((episode_length,), dtype=episode['timestamps'].dtype)
    transform_img = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop, 
                                                    out_resolutions_crop_random_ratio=out_resolutions_crop_random_ratio, 
                                                    bgr_to_rgb=True)
    obs_dict['rgb'] = ('image', f'{serial_id}_left', transform_img)
    if svo_stereo:
        obs_dict['rgb_right'] = ('image', f'{serial_id}_right', transform_img)
    if svo_depth:
        transform_depth = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop, is_depth=True)
        obs_dict['depth'] = ('depth', f'{serial_id}_left', transform_depth)
        if svo_stereo:
            obs_dict['depth_right'] = ('depth', f'{serial_id}_right', transform_depth)
    if svo_pointcloud:
        transform_pointcloud = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize, output_crop_res=out_resolutions_crop, is_depth=True)
        obs_dict['pointcloud'] = ('pointcloud', f'{serial_id}_left', transform_pointcloud)
    
    global_idx = 0
    next_global_idx = 0
    start_time = episode['timestamps'][0]
    for t in tqdm(range(frame_count), 'svo to episode'):
        # print(f"{t}: {next_global_idx}")
        svo_output = svo_camera.read_camera(return_timestamp=True)
        if svo_output is None:
            break
        else:
            data_dict, timestamp = svo_output
            timestamp = float(timestamp)
        if timestamp < episode['timestamps'][0] - dt:
            continue

        local_idxs, global_idxs, next_global_idx \
                = get_accumulate_timestamp_idxs(
                timestamps=[timestamp],
                start_time=start_time,
                dt=dt,
                next_global_idx=next_global_idx
            )

        if len(global_idxs) > 0:
            for global_idx in global_idxs:
                if global_idx == episode_length:
                    break
                for key in obs_dict.keys():
                    value = data_dict[obs_dict[key][0]][obs_dict[key][1]]
                    transform = obs_dict[key][2]
                    if value.shape[-1] == 4:
                        value = value[..., :3]
                    value = transform(value)
                    if 'rgb' in key and out_resolutions_image_final is not None:
                        value = cv2.resize(value, out_resolutions_image_final, interpolation=cv2.INTER_LINEAR)
                        value = value.astype(np.uint8)
                    if f'{camera_name}_{key}' not in episode.keys():
                        episode[f'{camera_name}_{key}'] = np.zeros((episode_length,) + value.shape, dtype=value.dtype)
                    episode[f'{camera_name}_{key}'][global_idx] = value
                episode[f'{camera_name}_real_timestamp'][global_idx] = timestamp
        if (next_global_idx == episode_length) or (global_idx == episode_length):
            break
        
    if (next_global_idx < episode_length) and (global_idx != episode_length):
        abandoned_frames = episode_length - next_global_idx
        for key in episode.keys():
            try:
                episode[key] = episode[key][:-abandoned_frames]
            except:
                pass
        print(f"Warning: {next_global_idx} < {episode_length}, abandoned {abandoned_frames} frames.")

    n_length = np.min([episode['timestamps'].shape[0], episode[f'{camera_name}_real_timestamp'].shape[0]])
    for key in episode.keys():
        episode[key] = episode[key][:n_length]

    return episode