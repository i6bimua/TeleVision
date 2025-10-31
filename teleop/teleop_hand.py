from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import math
import numpy as np
import torch

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
import os
import imageio.v2 as imageio

"""
6DOF 解释：
- 6 Degrees of Freedom（6 自由度）指观察者或物体在三维空间中的 6 个独立自由度：
  - 平移：沿 X、Y、Z 三个方向的位移（3DOF：x, y, z）
  - 旋转：绕 X、Y、Z 三个轴的转动（3DOF：roll, pitch, yaw）
- 在 360° 全景中：
  - 仅旋转（3DOF）可通过在一个大球内观看贴图实现，用户能朝任意方向看，但无法前后左右移动产生视差。
  - 完整 6DOF 需要体素/几何/深度等信息来支持位移视差；本文件方案A/方案B均为 3DOF 观看（可扩展）。
"""

class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir('../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)

        head_rmat = head_mat[:3, :3]

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos

class Sim:
    def __init__(self,
                 print_freq=False):
        self.print_freq = print_freq

        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)

        # load cube asset
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 10
        cube_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)

        asset_root = "../assets"
        left_asset_path = "inspire_hand/inspire_hand_left.urdf"
        right_asset_path = "inspire_hand/inspire_hand_right.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        self.dof = self.gym.get_asset_dof_count(left_asset)

        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # table
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.2)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', 0)
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # cube
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.25)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        cube_handle = self.gym.create_actor(self.env, cube_asset, pose, 'cube', 0)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # left_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.left_handle = self.gym.create_actor(self.env, left_asset, pose, 'left', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)

        # right_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.right_handle = self.gym.create_actor(self.env, right_asset, pose, 'right', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        self.left_root_states = self.root_states[left_idx]
        self.right_root_states = self.root_states[right_idx]

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([-0.6, 0, 1.6])

        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))

    def step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos):

        if self.print_freq:
            start = time.time()

        self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        left_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        left_states['pos'] = left_qpos
        self.gym.set_actor_dof_states(self.env, self.left_handle, left_states, gymapi.STATE_POS)

        right_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        right_states['pos'] = right_qpos
        self.gym.set_actor_dof_states(self.env, self.right_handle, right_states, gymapi.STATE_POS)

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)))
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)))
        left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))

        return left_image, right_image

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


class SimPano:
    def __init__(self, print_freq=False, out_dir="figures"):
        self.print_freq = print_freq
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # 场景与手模型与 Sim 一致（简化：不重复贴出全部道具）
        asset_root = "../assets"
        left_asset_path = "inspire_hand/inspire_hand_left.urdf"
        right_asset_path = "inspire_hand/inspire_hand_right.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)

        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.left_handle = self.gym.create_actor(self.env, left_asset, pose, 'left', 1, 1)
        self.right_handle = self.gym.create_actor(self.env, right_asset, pose, 'right', 1, 1)

        # 创建六个相机（±X, ±Y, ±Z）
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1024
        camera_props.height = 1024
        camera_props.horizontal_fov = 90.0
        self.cams = []
        self.cam_dirs = [
            (np.array([1.0, 0.0, 0.0]),  np.array([0.0, -1.0, 0.0])),  # +X
            (np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),  # -X
            (np.array([0.0, 1.0, 0.0]),  np.array([0.0, 0.0, 1.0])),   # +Y
            (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),   # -Y
            (np.array([0.0, 0.0, 1.0]),  np.array([0.0, -1.0, 0.0])),  # +Z
            (np.array([0.0, 0.0, -1.0]), np.array([0.0, -1.0, 0.0])),  # -Z
        ]
        self.eye = np.array([-0.6, 0.0, 1.6])
        for look, up in self.cam_dirs:
            h = self.gym.create_camera_sensor(self.env, camera_props)
            self._place_camera(h, self.eye, look, up)
            self.cams.append(h)

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

    def _place_camera(self, handle, eye_np, look_np, up_np):
        forward = look_np / (np.linalg.norm(look_np) + 1e-8)
        right = np.cross(forward, up_np)
        right = right / (np.linalg.norm(right) + 1e-8)
        up2 = np.cross(right, forward)
        R = np.stack([right, up2, forward], axis=1)  # columns: x(right), y(up), z(forward)
        q_wxyz = rotations.quaternion_from_matrix(R)
        q = gymapi.Quat(q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0])
        T = gymapi.Transform()
        T.p = gymapi.Vec3(float(eye_np[0]), float(eye_np[1]), float(eye_np[2]))
        T.r = q
        self.gym.set_camera_transform(self.env, handle, T)

    def _render_faces(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        faces = []
        for h in self.cams:
            img = self.gym.get_camera_image(self.sim, self.env, h, gymapi.IMAGE_COLOR)
            img = img.reshape(img.shape[0], -1, 4)[..., :3]
            faces.append(img)
        return faces  # 顺序与 self.cam_dirs 对应

    @staticmethod
    def _cubemap_to_equirect(faces, width=2048, height=1024):
        # 简易 CPU 版：将立方体贴图采样到 equirect，faces 顺序: +X,-X,+Y,-Y,+Z,-Z
        # 为简洁起见，这里使用最近邻选择，满足演示与流畅性；可后续换双线性插值。
        out = np.zeros((height, width, 3), dtype=np.uint8)
        # 生成经纬度网格
        theta = (np.linspace(0, width - 1, width) / width) * 2 * np.pi - np.pi  # [-pi, pi]
        phi = (np.linspace(0, height - 1, height) / height) * np.pi - (np.pi / 2)  # [-pi/2, pi/2]
        theta, phi = np.meshgrid(theta, phi)
        # 转方向向量
        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)
        vec = np.stack([x, y, z], axis=-1)

        def face_index_and_uv(v):
            ax = np.argmax(np.abs(v), axis=-1)
            i = ax == 0
            j = ax == 1
            k = ax == 2
            s = np.sign(v[np.arange(v.shape[0])[:, None], np.arange(v.shape[1])[None, :], ax])
            u = np.zeros_like(x)
            w = np.zeros_like(x)
            # +X/-X
            u[i] = -v[..., 2][i] / np.abs(v[..., 0][i])
            w[i] = v[..., 1][i] / np.abs(v[..., 0][i])
            # +Y/-Y
            u[j] = v[..., 0][j] / np.abs(v[..., 1][j])
            w[j] = v[..., 2][j] / np.abs(v[..., 1][j])
            # +Z/-Z
            u[k] = v[..., 0][k] / np.abs(v[..., 2][k])
            w[k] = -v[..., 1][k] / np.abs(v[..., 2][k])
            # 归一到 [0,1]
            uu = (u + 1) * 0.5
            vv = (w + 1) * 0.5
            # 选择面索引
            fi = np.zeros_like(ax)
            # +X(0) or -X(1)
            fi[i & (s > 0)] = 0
            fi[i & (s < 0)] = 1
            # +Y(2) or -Y(3)
            fi[j & (s > 0)] = 2
            fi[j & (s < 0)] = 3
            # +Z(4) or -Z(5)
            fi[k & (s > 0)] = 4
            fi[k & (s < 0)] = 5
            return fi, uu, vv

        fi, uu, vv = face_index_and_uv(vec)
        h_f, w_f, _ = faces[0].shape
        xi = np.clip((uu * (w_f - 1)).astype(np.int32), 0, w_f - 1)
        yi = np.clip((vv * (h_f - 1)).astype(np.int32), 0, h_f - 1)
        for idx in range(6):
            mask = fi == idx
            out[mask] = faces[idx][yi[mask], xi[mask]]
        return out

    def step(self):
        faces = self._render_faces()
        pano = self._cubemap_to_equirect(faces)
        imageio.imwrite(os.path.join(self.out_dir, 'pano.jpg'), pano, quality=85)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


class SimStereoPano(SimPano):
    def __init__(self, print_freq=False, out_dir="figures", ipd=0.066):
        super().__init__(print_freq=print_freq, out_dir=out_dir)
        self.ipd = ipd

    def _render_faces_eye(self, eye_offset):
        # 暂用重新定位相机方式（效率可优化为两套相机缓存）
        faces = []
        base = np.array([-0.6, 0.0, 1.6]) + eye_offset
        for (look, up), h in zip(self.cam_dirs, self.cams):
            self._place_camera(h, base, look, up)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        for h in self.cams:
            img = self.gym.get_camera_image(self.sim, self.env, h, gymapi.IMAGE_COLOR)
            img = img.reshape(img.shape[0], -1, 4)[..., :3]
            faces.append(img)
        return faces

    def step(self):
        left_faces = self._render_faces_eye(np.array([0.0, self.ipd * 0.5, 0.0]))
        right_faces = self._render_faces_eye(np.array([0.0, -self.ipd * 0.5, 0.0]))
        left_pano = self._cubemap_to_equirect(left_faces)
        right_pano = self._cubemap_to_equirect(right_faces)
        imageio.imwrite(os.path.join(self.out_dir, 'left_pano.jpg'), left_pano, quality=85)
        imageio.imwrite(os.path.join(self.out_dir, 'right_pano.jpg'), right_pano, quality=85)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sim', choices=['sim', 'pano', 'stereo_pano'], help='选择运行模式')
    args = parser.parse_args()

    if args.mode == 'sim':
        teleoperator = VuerTeleop('inspire_hand.yml')
        simulator = Sim()
        try:
            while True:
                head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
                left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
                np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))
        except KeyboardInterrupt:
            simulator.end()
            exit(0)

    elif args.mode == 'pano':
        # 方案A：单目360（equirectangular），写 figures/pano.jpg，TeleVision 用球内贴图展示
        # new teleophand (pano)
        # 这里不依赖手部重定向，仅渲染环境全景
        # 启动 Vuer 端（pano 场景）
        tv = OpenTeleVision((720, 1280), '', None, None, stream_mode='pano')
        simulator = SimPano(out_dir='figures')
        try:
            while True:
                simulator.step()
        except KeyboardInterrupt:
            simulator.end()
            exit(0)

    elif args.mode == 'stereo_pano':
        # 方案B：双目360（左右两张 equirectangular），写 figures/left_pano.jpg/right_pano.jpg
        # new teleophand (stereo pano)
        tv = OpenTeleVision((720, 1280), '', None, None, stream_mode='stereo_pano')
        simulator = SimStereoPano(out_dir='figures', ipd=0.066)
        try:
            while True:
                simulator.step()
        except KeyboardInterrupt:
            simulator.end()
            exit(0)

    elif args.mode == 'stereo_cubemap':
        # 方案B（天空盒）：双目 12 面立方体贴图
        # 仅生成 12 面图像；前端以 cubemap 方式加载（若不支持，则回退为双球体）
        tv = OpenTeleVision((720, 1280), '', None, None, stream_mode='stereo_cubemap')
        # 复用 SimStereoPano 的六面相机渲染，但直接保存面图
        sim = SimStereoPano(out_dir='figures', ipd=0.066)
        try:
            while True:
                # 渲染左右眼六面
                left_faces = sim._render_faces_eye(np.array([0.0, 0.033, 0.0]))
                right_faces = sim._render_faces_eye(np.array([0.0, -0.033, 0.0]))

                names = ['px','nx','py','ny','pz','nz']
                for img,name in zip(left_faces, names):
                    imageio.imwrite(os.path.join('figures', f'left_{name}.jpg'), img, quality=85)
                for img,name in zip(right_faces, names):
                    imageio.imwrite(os.path.join('figures', f'right_{name}.jpg'), img, quality=85)
                # 生成回退用的 equirectangular（供前端回退为双球体时使用）
                left_pano = sim._cubemap_to_equirect(left_faces)
                right_pano = sim._cubemap_to_equirect(right_faces)
                imageio.imwrite(os.path.join('figures', 'left_pano.jpg'), left_pano, quality=85)
                imageio.imwrite(os.path.join('figures', 'right_pano.jpg'), right_pano, quality=85)
                sim.gym.draw_viewer(sim.viewer, sim.sim, True)
                sim.gym.sync_frame_time(sim.sim)
        except KeyboardInterrupt:
            sim.end()
            exit(0)
