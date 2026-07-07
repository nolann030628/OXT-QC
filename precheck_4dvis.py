import argparse
from pathlib import Path
import time

from common.open3d_vis_utils import create_coordinate
from common.pose_utils import mat_to_pose
from common.pose_utils import pose_to_mat
from common.replay_buffer import ReplayBuffer
import imageio
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm
import viser
import viser.extras
import viser.transforms as tf

RIGHT_WRIST_COORD_ADJUST_MATRIX = None
LEFT_WRIST_COORD_ADJUST_MATRIX = None
EGO_CAMERA_ADJUST_MATRIX = None
DEFAULT_EXPORT_FPS = 15

if RIGHT_WRIST_COORD_ADJUST_MATRIX is None:
    RIGHT_WRIST_COORD_ADJUST_MATRIX = np.eye(4)
if LEFT_WRIST_COORD_ADJUST_MATRIX is None:
    LEFT_WRIST_COORD_ADJUST_MATRIX = np.eye(4)
if EGO_CAMERA_ADJUST_MATRIX is None:
    EGO_CAMERA_ADJUST_MATRIX = np.eye(4)


def _normalize_string_value(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8")
    return str(value)


def _sanitize_filename_part(value):
    value = _normalize_string_value(value)
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    sanitized = sanitized.strip("_")
    return sanitized or "unknown"


def _prepare_video_frames(frames):
    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis=-1)
    elif frames.ndim == 4:
        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        elif frames.shape[-1] == 4:
            frames = frames[..., :3]
        elif frames.shape[-1] != 3:
            raise ValueError(f"Unsupported video frame shape: {frames.shape}")
    else:
        raise ValueError(f"Expected video frames with shape (T, H, W) or (T, H, W, C), got {frames.shape}")

    if frames.dtype == np.uint8:
        return frames

    if frames.size == 0:
        return frames.astype(np.uint8)

    frames = np.nan_to_num(frames, nan=0.0, posinf=255.0, neginf=0.0)

    return np.clip(frames, 0, 255).astype(np.uint8)


def save_video(frames, output_path, fps=DEFAULT_EXPORT_FPS):
    frames = _prepare_video_frames(frames)
    if len(frames) == 0:
        raise ValueError(f"Cannot save empty video to {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=fps)


def _split_tactile_image_sequences(tactile_data):
    tactile_data = np.asarray(tactile_data)
    if tactile_data.ndim < 4:
        raise ValueError(f"image tactile expects at least 4 dims, got {tactile_data.shape}")

    if tactile_data.ndim == 4 and tactile_data.shape[-1] in (1, 3, 4):
        return [tactile_data]

    return [tactile_data[:, idx] for idx in range(tactile_data.shape[1])]


def _get_func_area_label(area_data, func_area_idx, num_func_areas, area_key):
    if area_data is None:
        return str(func_area_idx)

    area_data = np.asarray(area_data)
    if area_data.size == 0:
        return str(func_area_idx)

    if area_data.ndim == 0:
        return _normalize_string_value(area_data.item())

    if area_data.ndim == 1:
        if num_func_areas == 1:
            return _normalize_string_value(area_data[0])
        if area_data.shape[0] == num_func_areas:
            return _normalize_string_value(area_data[func_area_idx])
        return str(func_area_idx)

    if area_data.shape[1] <= func_area_idx:
        return str(func_area_idx)

    area_series = area_data[:, func_area_idx]
    area_value = area_series[0]
    if np.any(area_series != area_value):
        print(f"Warning: {area_key}[..., {func_area_idx}] changes over time; using first value {area_value}.")
    return _normalize_string_value(area_value)


def export_image_tactile_videos(episode, data_path, episode_idx, fps=DEFAULT_EXPORT_FPS):
    saved_paths = []
    data_path_stem = Path(str(data_path).rstrip("/")).stem or "episode"
    output_dir = Path("./vis_tactile_videos") / f"{data_path_stem}_ep{episode_idx}"

    tactile_data_keys = sorted(key for key in episode if "tactile_data" in key)
    for data_key in tactile_data_keys:
        type_key = data_key.replace("tactile_data", "tactile_type", 1)
        if type_key not in episode:
            continue

        tactile_types = np.asarray(episode[type_key]).reshape(-1)
        if tactile_types.size == 0:
            continue

        type_names = {_normalize_string_value(value).lower() for value in tactile_types}
        if type_names != {"image"}:
            continue

        try:
            func_area_frames_list = _split_tactile_image_sequences(episode[data_key])
        except ValueError as exc:
            print(f"Skip tactile video export for {data_key}: {exc}")
            continue

        area_key = data_key.replace("tactile_data", "tactile_area", 1)
        area_data = episode.get(area_key)

        for func_area_idx, func_area_frames in enumerate(func_area_frames_list):
            func_area_label = _sanitize_filename_part(
                _get_func_area_label(area_data, func_area_idx, len(func_area_frames_list), area_key)
            )
            output_path = output_dir / (
                f"{_sanitize_filename_part(data_key)}_func_area_{func_area_label}.mp4"
            )
            if output_path.exists():
                output_path = output_dir / (
                    f"{_sanitize_filename_part(data_key)}_func_area_{func_area_label}_slot_{func_area_idx}.mp4"
                )
            save_video(func_area_frames, output_path, fps=fps)
            saved_paths.append(output_path)
            print(f"Saved tactile video: {output_path.resolve()}")

    return saved_paths


def visualization_hand(server, point_nodes, frame_nodes, i, wrist_pose, finger_pos, is_right,
                       color=np.array([0, 0, 255]), point_size=0.01, axes_length=0.1, axes_radius=0.005):
    if wrist_pose is not None:
        wrist_frame = server.scene.add_frame(
            f"/frames/t{i}/wrist_{is_right}",
            position=wrist_pose[:3],
            axes_length=axes_length,
            axes_radius=axes_radius,
            wxyz=tf.SO3.from_matrix(Rotation.from_rotvec(
                wrist_pose[3:]).as_matrix()).wxyz
        )
        

    # use_idx = [0, 4, 5, 6, 7, 8]
    if finger_pos is not None:
        pts = []
        for idx in range(len(finger_pos)):
            pts.append(finger_pos[idx])
        # import pdb; pdb.set_trace()
        pts = np.array(pts)
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/hand_{is_right}",
                points=pts,
                colors=color,
                point_size=point_size,
                point_shape="rounded",
            ))


def visualization(
        videos,  # (T, H, W, C)
        intrinsic,  # (4, 4)
        pointclouds,  # List[(N, 6)], XYZ+RGB
        camera_poses,  # (T, 6)
        left_wrist_pose,  # (T, 6)
        right_wrist_pose,  # (T, 6)
        left_finger_pos,  # (T, N, 6)
        right_finger_pos,  # (T, N, 6)
        frustum_downsample_factor: int = 4,
        share: bool = False,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    num_frames = len(camera_poses)

    print("Start Visualization")
    camera_poses = pose_to_mat(camera_poses)


    if len(pointclouds) > 0:
        for i in range(num_frames):
            camera_p = camera_poses[i]
            pointclouds[i][:, :3] = (
                camera_p[:3, :3] @ pointclouds[i][:, :3, None])[:, :, 0] + camera_p[:3, 3]

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.001,
            max=0.02,
            step=1e-3,
            initial_value=0.005,
        )
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=15
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            # Toggle visibility.
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Add recording UI.
    with server.gui.add_folder("Recording"):
        gui_record_scene = server.gui.add_button("Record Scene")

    # Recording handler
    @gui_record_scene.on_click
    def _(_):
        gui_record_scene.disabled = True

        # Save the original frame visibility state
        original_visibility = [
            frame_node.visible for frame_node in frame_nodes]

        rec = server._start_scene_recording()
        rec.set_loop_start()

        # Determine sleep duration based on current FPS
        sleep_duration = 1.0 / \
            gui_framerate.value if gui_framerate.value > 0 else 0.033  # Default to ~30 FPS

        if gui_show_all_frames.value:
            # Record all frames according to the stride
            stride = gui_stride.value
            frames_to_record = [i for i in range(
                num_frames) if i % stride == 0]
        else:
            # Record the frames in sequence
            frames_to_record = range(num_frames)

        for t in frames_to_record:
            # Update the scene to show frame t
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i == t) if not gui_show_all_frames.value else (
                        i % gui_stride.value == 0)
            server.flush()
            rec.insert_sleep(sleep_duration)

        # set all invisible
        with server.atomic():
            for frame_node in frame_nodes:
                frame_node.visible = False

        # Finish recording
        bs = rec.end_and_serialize()

        # Save the recording to a file
        output_path = Path(
            f"./viser_result/recording_{str(data_path).split('/')[-1]}.viser")
        # make sure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bs)
        print(f"Recording saved to {output_path.resolve()}")

        # Restore the original frame visibility state
        with server.atomic():
            for frame_node, visibility in zip(frame_nodes, original_visibility):
                frame_node.visible = visibility
        server.flush()

        gui_record_scene.disabled = False

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([np.pi, 0.0, np.pi])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    point_nodes: list[viser.PointCloudHandle] = []

    for i in tqdm(range(num_frames)):
        if len(pointclouds) > 0:
            position, color = pointclouds[i][:, :3], pointclouds[i][:, 3:]
        rgb = videos[i]

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(
            f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        if len(pointclouds) > 0:
            point_nodes.append(
                server.scene.add_point_cloud(
                    name=f"/frames/t{i}/point_cloud",
                    points=position,
                    colors=color / 255.0,
                    point_size=gui_point_size.value,
                    point_shape="rounded",
                )
            )

        # Place the frustum.
        fov = 2 * np.arctan2(rgb.shape[0] / 2, intrinsic[0, 0])
        aspect = rgb.shape[1] / rgb.shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.05,
            image=rgb[::frustum_downsample_factor,
                      ::frustum_downsample_factor],
            wxyz=tf.SO3.from_matrix(camera_poses[i][:3, :3]).wxyz,
            position=camera_poses[i][:3, 3],
        )

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.1,
            axes_radius=0.005,
        )
        # wrist_pose, finger_pos
        left_wrist_pose_i = None if left_wrist_pose is None else left_wrist_pose[i]
        left_finger_pos_i = None if left_finger_pos is None else left_finger_pos[i]
        right_wrist_pose_i = right_wrist_pose[i]
        right_finger_pos_i = None if right_finger_pos is None else right_finger_pos[i]
        
        # 左wrist: 红色，较大的坐标系，较大的点，较大的标记球体
        visualization_hand(server, point_nodes, frame_nodes, i, left_wrist_pose_i, left_finger_pos_i, is_right=False,
                           color=np.array([255., 0., 0.]),  # 红色
                           point_size=0.01,  # 较大的点
                           axes_length=0.12,  # 较大的坐标系
                           axes_radius=0.006,  # 较粗的轴线
                           )  # 红色标记
        # 右wrist: 蓝色，较小的坐标系，较小的点，较小的标记球体
        visualization_hand(server, point_nodes, frame_nodes, i, right_wrist_pose_i, right_finger_pos_i, is_right=True,
                           color=np.array([0., 0., 255.]),  # 蓝色
                           point_size=0.0075,  # 较小的点
                           axes_length=0.08,  # 较小的坐标系
                           axes_radius=0.004,  # 较细的轴线
                           )  # 蓝色标记

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        # Update the timestep if we're playing.
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        # Update point size of both this timestep and the next one! There's
        # redundancy here, but this will be optimized out internally by viser.
        #
        # We update the point size for the next timestep so that it will be
        # immediately available when we toggle the visibility.
        if len(point_nodes) > 0:
            point_nodes[gui_timestep.value].point_size = gui_point_size.value
            point_nodes[
                (gui_timestep.value + 1) % num_frames
            ].point_size = gui_point_size.value

        time.sleep(1.0 / gui_framerate.value)


def main(args, max_points_distance=2.0):
    if args.data_path is None:
        raise ValueError("Please provide the path to the data directory.")

    if args.data_path.endswith(".zarr"):
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=args.data_path, mode='r')
        episode = replay_buffer.get_episode(args.episode_idx)
    else:
        raise NotImplementedError("Only support zarr format for now.")

    USE_LENGTH = args.use_length
    if USE_LENGTH is not None:
        for key in episode:
            if episode[key].shape[0] > USE_LENGTH:
                episode[key] = episode[key][:USE_LENGTH]

    left_wrist_pose = episode['left_wrist_pose'] if 'left_wrist_pose' in episode else None
    right_wrist_pose = episode['right_wrist_pose'] if 'right_wrist_pose' in episode else None
    left_finger_pose = episode['left_hand_pose'] if 'left_hand_pose' in episode else None
    right_finger_pose = episode['right_hand_pose'] if 'right_hand_pose' in episode else None
    left_finger_pos = left_finger_pose[..., :3] if left_finger_pose is not None else None
    right_finger_pos = right_finger_pose[..., :3] if right_finger_pose is not None else None


    if right_wrist_pose is not None:
        right_wrist_pose =  pose_to_mat(right_wrist_pose) @ RIGHT_WRIST_COORD_ADJUST_MATRIX
        right_wrist_pose = mat_to_pose(right_wrist_pose)

        # YZ_flip_mat = np.diag([-1, 1, 1, 1])
        # right_wrist_pose = mat_to_pose(YZ_flip_mat @ pose_to_mat(right_wrist_pose) @ YZ_flip_mat)

    if left_wrist_pose is not None:
        left_wrist_pose = pose_to_mat(left_wrist_pose) @ LEFT_WRIST_COORD_ADJUST_MATRIX
        left_wrist_pose = mat_to_pose(left_wrist_pose)

        
        # YZ_flip_mat = np.diag([-1, 1, 1, 1])
        # left_wrist_pose = mat_to_pose(YZ_flip_mat @ pose_to_mat(left_wrist_pose) @ YZ_flip_mat)

    if not args.disable_pointclouds:
        pointclouds = episode['camera_ego_pointcloud']    # (T, H, W, 3)
    if 'camera_ego_rgb' in episode:
        videos = episode['camera_ego_rgb']                    # (T, H, W, 3)
        if 'camera_ego_pose' in episode:
            camera_poses = episode['camera_ego_pose']
            camera_poses = pose_to_mat(camera_poses) @ EGO_CAMERA_ADJUST_MATRIX
            camera_poses = mat_to_pose(camera_poses)
        else:
            camera_poses = mat_to_pose(np.eye(4))
            camera_poses = np.stack([camera_poses] * len(videos), axis=0)
            print(camera_poses.shape)
    elif 'right_wrist_camera_rgb' in episode:
        robot_type = episode['right_hand_joints'].shape[-1] > 1
        videos = episode['right_wrist_camera_rgb']                    # (T, H, W, 3)
        camera_poses = np.stack([right_wrist_pose[0]] * len(videos), axis=0)
        if robot_type is False:
            # gripper robot type
            camera_adjust_mat = np.array([
                [0, -1, 0, 0],
                [1, 0, 0, 0.1],
                [0, 0, 1, 0.05],
                [0, 0, 0, 1]
            ])
            camera_adjust_mat = np.linalg.inv(camera_adjust_mat)
            camera_poses = mat_to_pose(pose_to_mat(camera_poses) @ camera_adjust_mat)
        else:
            pass
    else:
        raise ValueError("No camera RGB data found in the episode.")

    if args.disable_pointclouds:
        print("Disable pointclouds visualization.")
        pointclouds_vis = []
    else:
        pointclouds_vis = []
        for t in tqdm(range(len(pointclouds)), desc="PointClouds-Process"):
            useful = (~np.isinf(pointclouds[t])) & (~np.isnan(pointclouds[t]))
            useful = (useful.sum(axis=-1) == 3.0)
            xyz, rgb = pointclouds[t][useful], videos[t][useful]
            xyz, rgb = xyz.reshape(-1, 3), rgb.reshape(-1, 3)
            distance = np.linalg.norm(xyz, axis=-1)
            rgb = rgb[distance < max_points_distance]
            xyz = xyz[distance < max_points_distance]
            rgb = rgb[::args.downsample_factor]
            xyz = xyz[::args.downsample_factor]
            pointclouds_vis.append(np.concatenate(
                [xyz, rgb], axis=-1))   # (N, 6)
            # break

    if args.precheck_pointclouds and not args.disable_pointclouds:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Data Visualizer, Press Q to Exist", width=800, height=600)
        pcd_xyz, pcd_rgb = [], []
        coord = create_coordinate(np.zeros(3), np.eye(3), size=0.4)
        vis.add_geometry(coord)
        interval = len(pointclouds_vis) // 4

        sphere_org = create_coordinate(np.zeros(3), np.eye(3), size=0.2)
        vis.add_geometry(sphere_org)

        for i, pcd in enumerate(pointclouds_vis):

            if interval > 0 and i % interval > 0:
                continue

            camera_pose = pose_to_mat(camera_poses[i])

            coord = create_coordinate(
                camera_pose[:3, 3], camera_pose[:3, :3], size=0.05 + 0.1 * (i / (len(pointclouds_vis) - 1)))
            # coord = create_coordinate(camera_pose[:3, 3], camera_pose[:3, :3], size=0.15)
            vis.add_geometry(coord)

            print(f"pos{i}", camera_pose[:3, 3])
            print(f"rot{i}", Rotation.from_matrix(
                camera_pose[:3, :3]).as_euler("xyz", degrees=True))
            if True:
                print("Frame", i)
                xyz, rgb = pointclouds_vis[i][...,
                                              :3], pointclouds_vis[i][..., 3:]
                xyz = (camera_pose[:3, :3] @ xyz[:, :, None]
                       )[:, :, 0] + camera_pose[:3, 3]
                pcd_xyz.append(xyz)
                pcd_rgb.append(rgb)
        pcd_xyz = np.concatenate(pcd_xyz, axis=0)
        pcd_rgb = np.concatenate(pcd_rgb, axis=0)
        print(pcd_xyz.shape, pcd_rgb.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb / 255)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

    save_video(videos, "vis_videos.mp4")
    export_image_tactile_videos(episode, args.data_path, args.episode_idx)

    # a random intrinsic matrix
    intrinsic = np.array([
        [640, 0, 320],
        [0, 480, 240],
        [0, 0, 1]
    ])

    visualization(videos, intrinsic, pointclouds_vis, camera_poses, left_wrist_pose, right_wrist_pose, left_finger_pos,
                  right_finger_pos)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Visualize QuestRecord data.")
    arg_parser.add_argument("--data_path", type=str, help="Path to the data directory.")
    arg_parser.add_argument("-l", "--use_length", type=int, default=None, help="Use length of the data.")
    arg_parser.add_argument("-i", "--episode_idx", type=int, default=0, help="Episode index.")
    arg_parser.add_argument("--downsample_factor", type=int, default=4,
                            help="Downsample factor for rgb/depth visualization.")
    arg_parser.add_argument("-d", "--disable_pointclouds", action="store_true", help="Also visualization pointclouds.")
    arg_parser.add_argument("-p", "--precheck_pointclouds", action="store_true", help="Also visualization pointclouds.")
    args = arg_parser.parse_args()

    main(args)

# Example:
# python precheck_4dvis.py -d --data_path FTP1_data_example\\MotionTrans\\robot_se+put_bread_to_the_red_pad+.zarr
# python precheck_4dvis.py -d --data_path FTP1_data_example\\ViTaMIn\\peg_insertion.zarr