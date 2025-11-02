from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import math
import numpy as np
import torch

from TeleVision_panorama import OpenTeleVision
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
from datetime import datetime

# Import panorama conversion utilities
from panorama_utils import PanoramaConverter

# Import py360convert for fallback conversion
try:
    import py360convert
    PY360CONVERT_AVAILABLE = True
except ImportError:
    # print("Warning: py360convert not installed. Install with: pip install py360convert")
    PY360CONVERT_AVAILABLE = False

# Import OpenCV for acceleration (compatible with Python 3.8)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class LatencyMonitor:
    """延迟监控类，用于追踪整个pipeline的计算延迟和传输延迟"""
    def __init__(self, log_file_path="latency_log.txt"):
        self.log_file_path = log_file_path
        self.frame_timings = []
        self.current_frame = {}
        self.frame_count = 0
        
        # 创建日志文件并写入头部
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 延迟监控日志 - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def start_frame(self):
        """开始新的一帧，记录总开始时间"""
        self.frame_count += 1
        self.current_frame = {
            'frame_id': self.frame_count,
            'start_time': time.time(),
            'compute_delays': {},
            'transmit_delays': {},
            'timestamps': {}
        }
        self.current_frame['timestamps']['frame_start'] = self.current_frame['start_time']
    
    def record_compute(self, name, start_time, end_time):
        """记录计算延迟"""
        delay = (end_time - start_time) * 1000  # 转换为毫秒
        self.current_frame['compute_delays'][name] = delay
        self.current_frame['timestamps'][f'{name}_end'] = end_time
    
    def record_transmit(self, name, start_time, end_time):
        """记录传输延迟"""
        delay = (end_time - start_time) * 1000  # 转换为毫秒
        self.current_frame['transmit_delays'][name] = delay
        self.current_frame['timestamps'][f'{name}_end'] = end_time
    
    def end_frame(self):
        """结束当前帧，计算总延迟和帧率，写入文件"""
        end_time = time.time()
        total_delay = (end_time - self.current_frame['start_time']) * 1000  # 总延迟（毫秒）
        fps = 1000.0 / total_delay if total_delay > 0 else 0
        
        self.current_frame['total_delay_ms'] = total_delay
        self.current_frame['fps'] = fps
        self.current_frame['end_time'] = end_time
        self.current_frame['timestamps']['frame_end'] = end_time
        
        # 写入文件
        self._write_frame_to_file()
        
        # 保存到列表用于统计分析
        self.frame_timings.append(self.current_frame.copy())
    
    def _write_frame_to_file(self):
        """将当前帧的延迟信息写入文件（实际使用中已禁用文件写入）"""
        # 实际使用中已禁用文件写入，减少I/O开销
        # with open(self.log_file_path, 'a', encoding='utf-8') as f:
        #     f.write(f"\n--- 帧 #{self.current_frame['frame_id']} ---\n")
        #     f.write(f"总延迟: {self.current_frame['total_delay_ms']:.3f} ms, 帧率: {self.current_frame['fps']:.2f} FPS\n\n")
        #     
        #     f.write("计算延迟:\n")
        #     for name, delay in self.current_frame['compute_delays'].items():
        #         f.write(f"  {name}: {delay:.3f} ms\n")
        #     
        #     f.write("\n传输延迟:\n")
        #     for name, delay in self.current_frame['transmit_delays'].items():
        #         f.write(f"  {name}: {delay:.3f} ms\n")
        #     
        #     f.write("\n")
        pass
    
    def generate_summary(self):
        """生成统计摘要并写入文件"""
        if len(self.frame_timings) == 0:
            return
        
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"=== 统计分析 - 总帧数: {len(self.frame_timings)} ===\n")
            f.write(f"{'='*80}\n\n")
            
            # 收集所有延迟项
            all_compute_keys = set()
            all_transmit_keys = set()
            for frame in self.frame_timings:
                all_compute_keys.update(frame['compute_delays'].keys())
                all_transmit_keys.update(frame['transmit_delays'].keys())
            
            # 计算平均值
            f.write("=== 计算延迟统计 ===\n")
            compute_stats = {}
            for key in all_compute_keys:
                values = [f['compute_delays'].get(key, 0) for f in self.frame_timings if key in f['compute_delays']]
                if values:
                    compute_stats[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'total': np.sum(values)
                    }
                    f.write(f"{key}:\n")
                    f.write(f"  平均值: {compute_stats[key]['mean']:.3f} ms\n")
                    f.write(f"  标准差: {compute_stats[key]['std']:.3f} ms\n")
                    f.write(f"  最小值: {compute_stats[key]['min']:.3f} ms\n")
                    f.write(f"  最大值: {compute_stats[key]['max']:.3f} ms\n")
                    f.write(f"  累计: {compute_stats[key]['total']:.3f} ms\n\n")
            
            f.write("\n=== 传输延迟统计 ===\n")
            transmit_stats = {}
            for key in all_transmit_keys:
                values = [f['transmit_delays'].get(key, 0) for f in self.frame_timings if key in f['transmit_delays']]
                if values:
                    transmit_stats[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'total': np.sum(values)
                    }
                    f.write(f"{key}:\n")
                    f.write(f"  平均值: {transmit_stats[key]['mean']:.3f} ms\n")
                    f.write(f"  标准差: {transmit_stats[key]['std']:.3f} ms\n")
                    f.write(f"  最小值: {transmit_stats[key]['min']:.3f} ms\n")
                    f.write(f"  最大值: {transmit_stats[key]['max']:.3f} ms\n")
                    f.write(f"  累计: {transmit_stats[key]['total']:.3f} ms\n\n")
            
            # 总延迟统计
            total_delays = [f['total_delay_ms'] for f in self.frame_timings]
            fps_list = [f['fps'] for f in self.frame_timings]
            
            f.write("\n=== 总体延迟统计 ===\n")
            f.write(f"总延迟平均值: {np.mean(total_delays):.3f} ms\n")
            f.write(f"总延迟标准差: {np.std(total_delays):.3f} ms\n")
            f.write(f"总延迟最小值: {np.min(total_delays):.3f} ms\n")
            f.write(f"总延迟最大值: {np.max(total_delays):.3f} ms\n\n")
            
            f.write("\n=== 帧率统计 ===\n")
            f.write(f"平均帧率: {np.mean(fps_list):.2f} FPS\n")
            f.write(f"帧率标准差: {np.std(fps_list):.2f} FPS\n")
            f.write(f"最低帧率: {np.min(fps_list):.2f} FPS\n")
            f.write(f"最高帧率: {np.max(fps_list):.2f} FPS\n\n")
            
            # 延迟占比分析
            if compute_stats or transmit_stats:
                f.write("\n=== 延迟占比分析 ===\n")
                total_compute_time = sum([s['total'] for s in compute_stats.values()])
                total_transmit_time = sum([s['total'] for s in transmit_stats.values()])
                total_frame_time = sum(total_delays)
                
                f.write(f"总计算时间: {total_compute_time:.3f} ms ({total_compute_time/total_frame_time*100:.2f}%)\n")
                f.write(f"总传输时间: {total_transmit_time:.3f} ms ({total_transmit_time/total_frame_time*100:.2f}%)\n")
                f.write(f"总帧时间: {total_frame_time:.3f} ms\n")
                
                # 各模块占比
                f.write("\n各计算模块占比:\n")
                for key, stats in compute_stats.items():
                    percentage = stats['total'] / total_frame_time * 100
                    f.write(f"  {key}: {percentage:.2f}%\n")
                
                f.write("\n各传输模块占比:\n")
                for key, stats in transmit_stats.items():
                    percentage = stats['total'] / total_frame_time * 100
                    f.write(f"  {key}: {percentage:.2f}%\n")
            
            f.write(f"\n{'='*80}\n")
            f.write(f"=== 分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")


class VuerTeleop:
    def __init__(self, config_file_path, latency_monitor=None):
        # 全景图分辨率：2:1的宽高比，高度为原分辨率
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)
        
        # 全景图的宽高比为2:1
        self.panorama_height = self.resolution_cropped[0]  # 720
        self.panorama_width = self.panorama_height * 2     # 1440
        
        self.img_shape = (self.panorama_height, self.panorama_width, 3)
        self.img_height, self.img_width = self.panorama_height, self.panorama_width

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        # 使用全景模式：传入全景图尺寸而不是原分辨率
        self.tv = OpenTeleVision(self.img_shape, self.shm.name, image_queue, toggle_streaming, ngrok=True, panorama_mode=True)
        self.processor = VuerPreprocessor()
        self.latency_monitor = latency_monitor

        # 获取当前脚本所在目录的父目录（项目根目录）
        project_root = Path(__file__).parent.parent
        assets_dir = project_root / 'assets'
        RetargetingConfig.set_default_urdf_dir(str(assets_dir))
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        # 从预处理器获取处理后的姿态数据
        # head_mat: (4, 4) - 头部齐次变换矩阵
        # left_wrist_mat: (4, 4) - 左手腕齐次变换矩阵（已转换为Z-UP坐标系，相对头部）
        # right_wrist_mat: (4, 4) - 右手腕齐次变换矩阵（已转换为Z-UP坐标系，相对头部）
        # left_hand_mat: (25, 3) - 左手25个关键点的3D坐标（已转换为相对手腕的Inspire Hand坐标系）
        # right_hand_mat: (25, 3) - 右手25个关键点的3D坐标（已转换为相对手腕的Inspire Hand坐标系）
        
        # 监控processor处理延迟（包含从共享内存读取数据+计算的延迟）
        # 注意：processor.process()内部会从共享内存读取数据，这部分时间包含在计算延迟中
        if self.latency_monitor:
            t_process_start = time.time()
        
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
        
        if self.latency_monitor:
            t_process_end = time.time()
            self.latency_monitor.record_compute("teleop_preprocessor", t_process_start, t_process_end)

        # 提取头部旋转矩阵（3×3）
        if self.latency_monitor:
            t_extract_start = time.time()
        
        head_rmat = head_mat[:3, :3]  # shape: (3, 3)
        
        if self.latency_monitor:
            t_extract_end = time.time()
            self.latency_monitor.record_compute("teleop_extract_head_rmat", t_extract_start, t_extract_end)

        # 构建左手姿态: [x, y, z, qx, qy, qz, qw]
        # left_wrist_mat[:3, 3]: (3,) - 位置向量，加上固定偏移 [-0.6, 0, 1.6]
        # rotations.quaternion_from_matrix(): 返回 [w, x, y, z]，重排为 [x, y, z, w]
        if self.latency_monitor:
            t_left_pose_start = time.time()
        
        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),  # (3,) 位置
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])  # (4,) 四元数
        # left_pose shape: (7,)
        
        if self.latency_monitor:
            t_left_pose_end = time.time()
            self.latency_monitor.record_compute("teleop_build_left_pose", t_left_pose_start, t_left_pose_end)

        # 构建右手姿态: [x, y, z, qx, qy, qz, qw]
        if self.latency_monitor:
            t_right_pose_start = time.time()
        
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),  # (3,) 位置
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])  # (4,) 四元数
        # right_pose shape: (7,)
        
        if self.latency_monitor:
            t_right_pose_end = time.time()
            self.latency_monitor.record_compute("teleop_build_right_pose", t_right_pose_start, t_right_pose_end)

        # 手指重定向：从5个指尖位置推断12个关节角度
        # left_hand_mat[tip_indices]: (5, 3) - 5个指尖位置（tip_indices = [4, 9, 14, 19, 24]）
        # retarget() 返回: (12,) - 12个关节角度
        # [[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]: 重排关节顺序以匹配URDF定义
        if self.latency_monitor:
            t_retarget_start = time.time()
        
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]  # shape: (12,)
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]  # shape: (12,)
        
        if self.latency_monitor:
            t_retarget_end = time.time()
            self.latency_monitor.record_compute("teleop_hand_retargeting", t_retarget_start, t_retarget_end)

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos
        # 返回:
        #   head_rmat: (3, 3) - 头部旋转矩阵
        #   left_pose: (7,) - 左手位置+姿态 [x, y, z, qx, qy, qz, qw]
        #   right_pose: (7,) - 右手位置+姿态 [x, y, z, qx, qy, qz, qw]
        #   left_qpos: (12,) - 左手12个关节角度
        #   right_qpos: (12,) - 右手12个关节角度


class Sim:
    def __init__(self,
                 print_freq=False,
                 use_panorama=True,
                 latency_monitor=None):
        self.print_freq = print_freq
        self.use_panorama = use_panorama
        self.latency_monitor = latency_monitor
        
        # 初始化全景图转换器（包含缓存和优化）
        self.panorama_converter = PanoramaConverter()

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

        # 获取assets目录的绝对路径
        asset_root = str(Path(__file__).parent.parent / 'assets')
        left_asset_path = "inspire_hand/inspire_hand_left.urdf"
        right_asset_path = "inspire_hand/inspire_hand_right.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
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

        self.cam_pos = np.array([-0.6, 0, 1.6])

        # 设置全景相机系统
        if self.use_panorama:
            self._setup_panorama_cameras()
        else:
            # 旧的左右相机设置（保持兼容性）
            self.cam_lookat_offset = np.array([1, 0, 0])
            self.left_cam_offset = np.array([0, 0.033, 0])
            self.right_cam_offset = np.array([0, -0.033, 0])
            
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 720
            self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
            self.gym.set_camera_location(self.left_camera_handle,
                                         self.env,
                                         gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                         gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 720
            self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
            self.gym.set_camera_location(self.right_camera_handle,
                                         self.env,
                                         gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                         gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))

    def _setup_panorama_cameras(self):
        """设置六个面的相机 - 使用与OLD完全相同的方式"""
        # 每个面的分辨率 - 使用与VuerTeleop一致的全景图分辨率
        # VuerTeleop使用720高度，所以每面也是720
        self.face_resolution = 720  # 与VuerTeleop一致
        self.panorama_height = self.face_resolution  # 全景图高度等于每面高度
        self.panorama_width = self.panorama_height * 2  # 全景图宽高比为2:1
        
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.face_resolution
        camera_props.height = self.face_resolution
        camera_props.horizontal_fov = 90.0  # 90度FOV对应立方体的一个面
        
        # 定义六个面的lookat offset - 使用与OLD相同的结构
        self.cam_lookat_offset_F = np.array([1, 0, 0])   # Front: 看向+X
        self.cam_lookat_offset_R = np.array([0, -1, 0])   # Right: 看向-Y (右侧)
        self.cam_lookat_offset_B = np.array([-1, 0, 0])  # Back: 看向-X
        self.cam_lookat_offset_L = np.array([0, 1, 0])  # Left: 看向-Y (左侧)
        self.cam_lookat_offset_U = np.array([0.000001, 0, 1])   # Up: 看向+Z
        self.cam_lookat_offset_D = np.array([0.000001, 0, -1])  # Down: 看向-Z
        
        # 创建六个相机，并使用与OLD完全相同的方式设置
        self.cube_camera_handles = {}
        
        # Front camera
        self.cube_camera_handles['F'] = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.cube_camera_handles['F'],
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_F)))
        
        # Right camera
        self.cube_camera_handles['R'] = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.cube_camera_handles['R'],
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_R)))
        
        # Back camera
        self.cube_camera_handles['B'] = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.cube_camera_handles['B'],
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_B)))
        
        # Left camera
        self.cube_camera_handles['L'] = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.cube_camera_handles['L'],
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_L)))
        
        # Up camera
        self.cube_camera_handles['U'] = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.cube_camera_handles['U'],
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_U)))
        
        # Down camera
        self.cube_camera_handles['D'] = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.cube_camera_handles['D'],
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_D)))

    def step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos):
        # 调试：帧率统计（实际使用中可注释掉）
        # if self.print_freq:
        #     start = time.time()

        # 设置actor状态（计算+传输延迟）
        if self.latency_monitor:
            t_set_state_start = time.time()
        
        self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        
        if self.latency_monitor:
            t_set_state_end = time.time()
            self.latency_monitor.record_transmit("sim_set_actor_root_state", t_set_state_start, t_set_state_end)

        if self.latency_monitor:
            t_set_dof_start = time.time()
        
        left_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        left_states['pos'] = left_qpos
        self.gym.set_actor_dof_states(self.env, self.left_handle, left_states, gymapi.STATE_POS)

        right_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        right_states['pos'] = right_qpos
        self.gym.set_actor_dof_states(self.env, self.right_handle, right_states, gymapi.STATE_POS)
        
        if self.latency_monitor:
            t_set_dof_end = time.time()
            self.latency_monitor.record_transmit("sim_set_actor_dof_state", t_set_dof_start, t_set_dof_end)

        # step the physics（计算延迟）
        if self.latency_monitor:
            t_physics_start = time.time()
        
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        if self.latency_monitor:
            t_physics_end = time.time()
            self.latency_monitor.record_compute("sim_physics_simulation", t_physics_start, t_physics_end)

        # 相机捕获和转换（计算+传输延迟）
        if self.use_panorama:
            panorama_image = self._capture_and_convert_panorama(head_rmat)
        else:
            # 旧的相机捕获逻辑
            if self.latency_monitor:
                t_cam_setup_start = time.time()
            
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
            
            if self.latency_monitor:
                t_cam_setup_end = time.time()
                self.latency_monitor.record_compute("sim_camera_setup", t_cam_setup_start, t_cam_setup_end)
            
            if self.latency_monitor:
                t_cam_capture_start = time.time()
            
            left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
            right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
            
            if self.latency_monitor:
                t_cam_capture_end = time.time()
                self.latency_monitor.record_transmit("sim_camera_capture", t_cam_capture_start, t_cam_capture_end)
            
            if self.latency_monitor:
                t_img_process_start = time.time()
            
            left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
            right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]
            panorama_image = np.hstack((left_image, right_image))
            
            if self.latency_monitor:
                t_img_process_end = time.time()
                self.latency_monitor.record_compute("sim_image_processing", t_img_process_start, t_img_process_end)

        # 渲染（计算延迟）
        if self.latency_monitor:
            t_render_start = time.time()
        
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)
        
        if self.latency_monitor:
            t_render_end = time.time()
            self.latency_monitor.record_compute("sim_viewer_render", t_render_start, t_render_end)

        # 调试：打印帧率（实际使用中可注释掉）
        # if self.print_freq:
        #     end = time.time()
        #     print('Frequency:', 1 / (end - start))

        return panorama_image

    def _capture_and_convert_panorama(self, head_rmat=None):
        """捕获六个面并转换为全景图 - 相机朝向固定，不跟随头部姿态"""
        # 使用固定的相机朝向，不应用头部旋转
        # 使用完全相同的函数调用，但使用原始的lookat offset
        if self.latency_monitor:
            t_cam_setup_start = time.time()
        
        self.gym.set_camera_location(self.cube_camera_handles['F'], self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_F)))
        self.gym.set_camera_location(self.cube_camera_handles['R'], self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_R)))
        self.gym.set_camera_location(self.cube_camera_handles['B'], self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_B)))
        self.gym.set_camera_location(self.cube_camera_handles['L'], self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_L)))
        self.gym.set_camera_location(self.cube_camera_handles['U'], self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_U)))
        self.gym.set_camera_location(self.cube_camera_handles['D'], self.env,
                                     gymapi.Vec3(*(self.cam_pos)),
                                     gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_D)))
        
        if self.latency_monitor:
            t_cam_setup_end = time.time()
            self.latency_monitor.record_compute("panorama_camera_setup", t_cam_setup_start, t_cam_setup_end)
        
        # 捕获六个面的图像
        if self.latency_monitor:
            t_capture_start = time.time()
        
        face_order = ['F', 'R', 'B', 'L', 'U', 'D']
        cube_faces = []
        
        for face_name in face_order:
            handle = self.cube_camera_handles[face_name]
            img = self.gym.get_camera_image(self.sim, self.env, handle, gymapi.IMAGE_COLOR)
            # 转换为RGB格式
            img = img.reshape(img.shape[0], -1, 4)[..., :3]
            # Right和Back相机图像需要镜像反转
            if face_name in ['R', 'B']:
                img = np.fliplr(img)
            cube_faces.append(img)
        
        if self.latency_monitor:
            t_capture_end = time.time()
            self.latency_monitor.record_transmit("panorama_camera_capture", t_capture_start, t_capture_end)
        
        # 转换为horizon格式（使用工具函数）
        if self.latency_monitor:
            t_horizon_start = time.time()
        
        cube_horizon = self.panorama_converter.convert_cube_faces_to_horizon(cube_faces)
        
        if self.latency_monitor:
            t_horizon_end = time.time()
            self.latency_monitor.record_compute("panorama_horizon_conversion", t_horizon_start, t_horizon_end)
        
        # 转换为全景图
        if PY360CONVERT_AVAILABLE:
            if self.latency_monitor:
                t_convert_start = time.time()
            
            # 优先使用OpenCV加速版本（兼容Python 3.8）
            if OPENCV_AVAILABLE:
                panorama_uint8 = self.panorama_converter.convert_cube_horizon_to_equirectangular(
                    cube_horizon, self.panorama_height, self.panorama_width
                )
            else:
                # 使用原始py360convert版本（较慢）
                panorama = py360convert.c2e(
                    cube_horizon,
                    self.panorama_height,
                    self.panorama_width,
                    mode='bilinear',
                    cube_format='horizon'
                )
                # py360convert返回float64 [0-255]，需要转换为uint8
                panorama_uint8 = panorama.astype(np.uint8)
            
            if self.latency_monitor:
                t_convert_end = time.time()
                self.latency_monitor.record_compute("panorama_equirectangular_conversion", t_convert_start, t_convert_end)
            
            return panorama_uint8
        else:
            # 如果没有py360convert，返回一个占位符
            # print("Warning: py360convert not available, returning placeholder")
            return np.zeros((self.panorama_height, self.panorama_width, 3), dtype=np.uint8)

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-panorama', action='store_true', help='Use old stereo camera setup instead of panorama')
    parser.add_argument('--print-freq', action='store_true', help='Print frame rate')
    parser.add_argument('--latency-log', type=str, default='latency_log.txt', help='Path to latency log file')
    args = parser.parse_args()
    
    # 创建延迟监控器（实际使用中已禁用，如需启用延迟监控请取消注释）
    # latency_monitor = LatencyMonitor(log_file_path=args.latency_log)
    latency_monitor = None
    
    # 获取配置文件路径（相对于脚本所在目录）
    config_path = Path(__file__).parent / 'inspire_hand.yml'
    teleoperator = VuerTeleop(str(config_path), latency_monitor=latency_monitor)
    simulator = Sim(print_freq=args.print_freq, use_panorama=not args.no_panorama, latency_monitor=latency_monitor)

    frame_count = 0
    try:
        while True:
            # 开始新的一帧（延迟监控已禁用）
            # if latency_monitor:
            #     latency_monitor.start_frame()
            
            # Teleoperator步骤
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            
            # Simulator步骤
            panorama_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
            
            # 图像传输延迟（写入共享内存）（延迟监控已禁用）
            # if latency_monitor:
            #     t_img_transmit_start = time.time()
            
            np.copyto(teleoperator.img_array, panorama_img)
            
            # if latency_monitor:
            #     t_img_transmit_end = time.time()
            #     latency_monitor.record_transmit("image_shared_memory_write", t_img_transmit_start, t_img_transmit_end)
            
            # 结束当前帧，写入延迟信息（延迟监控已禁用）
            # if latency_monitor:
            #     latency_monitor.end_frame()
            
            # 调试：保存前几帧全景图（已禁用）
            # if frame_count < 3:
            #     from PIL import Image
            #     debug_img = Image.fromarray(panorama_img, 'RGB')
            #     debug_img.save(f'teleop/panorama_debug_{frame_count}.png')
            #     print(f"Debug: Saved panorama frame {frame_count} to teleop/panorama_debug_{frame_count}.png")
            
            frame_count += 1
    except KeyboardInterrupt:
        # 调试：生成延迟统计摘要（实际使用中已禁用）
        # if latency_monitor:
        #     latency_monitor.generate_summary()
        simulator.end()
        exit(0)

