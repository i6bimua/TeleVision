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

# 获取项目根目录（TeleVision）- 无论从哪里运行都能正确解析
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # teleop 的父目录就是项目根目录
_ASSETS_DIR = _PROJECT_ROOT / 'assets'

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
    """延迟监控类，用于追踪整个pipeline的计算延迟、传输延迟和带宽不足导致的延迟"""
    def __init__(self, log_file_path="latency_log_panorama_local.txt", network_mode=False):
        self.log_file_path = log_file_path
        self.frame_timings = []
        self.current_frame = {}
        self.frame_count = 0
        self.network_mode = network_mode
        
        # 创建日志文件并写入头部
        mode_str = "网络模式" if network_mode else "本地模式"
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Panorama Pipeline延迟监控日志 ({mode_str}) - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
            f.write("监控内容：\n")
            f.write("  - 计算延迟：各模块的计算时间\n")
            f.write("  - 传输延迟：数据读写、共享内存、网络传输等\n")
            f.write("  - 带宽延迟：带宽不足导致的队列阻塞、传输延迟等\n\n")
            if network_mode:
                f.write("注意：网络模式下还包括以下延迟：\n")
                f.write("  - VR设备输入延迟 (通常 5-10ms)\n")
                f.write("  - 网络传输延迟 (VR->主机, 通常 10-50ms，取决于网络质量)\n")
                f.write("  - 网络传输延迟 (主机->VR, 通常 10-50ms)\n")
                f.write("  - VR设备显示延迟 (通常 10-20ms)\n")
                f.write("体感延迟 ≈ 计算延迟 + 网络往返延迟 + VR设备延迟 (通常 +30-100ms)\n\n")
    
    def start_frame(self):
        """开始新的一帧，记录总开始时间"""
        self.frame_count += 1
        self.current_frame = {
            'frame_id': self.frame_count,
            'start_time': time.time(),
            'compute_delays': {},
            'transmit_delays': {},
            'bandwidth_delays': {},
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
    
    def record_bandwidth(self, name, delay_ms, extra_info=None):
        """记录带宽不足导致的延迟"""
        self.current_frame['bandwidth_delays'][name] = delay_ms
        if extra_info:
            self.current_frame[f'{name}_info'] = extra_info
    
    def end_frame(self, end_time=None):
        """结束当前帧，计算总延迟和帧率，写入文件
        
        Args:
            end_time: 可选的结束时间。如果为None，使用当前时间。
                      用于panorama模式中排除sleep时间。
        """
        if end_time is None:
            end_time = time.time()
        total_delay = (end_time - self.current_frame['start_time']) * 1000  # 总延迟（毫秒）
        fps = 1000.0 / total_delay if total_delay > 0 else 0
        
        self.current_frame['total_delay_ms'] = total_delay
        self.current_frame['fps'] = fps
        self.current_frame['end_time'] = end_time
        self.current_frame['timestamps']['frame_end'] = end_time
        
        # 验证：计算所有子延迟的总和
        total_compute = sum(self.current_frame['compute_delays'].values())
        total_transmit = sum(self.current_frame['transmit_delays'].values())
        total_bandwidth = sum(self.current_frame.get('bandwidth_delays', {}).values())
        total_sub_delays = total_compute + total_transmit + total_bandwidth
        
        # 计算未测量的时间（间隙时间）
        unmeasured_time = total_delay - total_sub_delays
        self.current_frame['unmeasured_time_ms'] = unmeasured_time
        self.current_frame['total_sub_delays_ms'] = total_sub_delays
        
        # 如果未测量时间超过总延迟的5%，发出警告（但不写入文件，避免干扰）
        if abs(unmeasured_time) > total_delay * 0.05:
            self.current_frame['_warning_large_gap'] = True
        
        # 写入文件
        self._write_frame_to_file()
        
        # 保存到列表用于统计分析
        self.frame_timings.append(self.current_frame.copy())
    
    def _write_frame_to_file(self):
        """将当前帧的延迟信息写入文件"""
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- 帧 #{self.current_frame['frame_id']} ---\n")
            f.write(f"总延迟（主机处理）: {self.current_frame['total_delay_ms']:.3f} ms, 帧率: {self.current_frame['fps']:.2f} FPS\n")
            
            # 显示闭环延迟（如果存在）
            if 'full_loop_delay_ms' in self.current_frame:
                full_loop_delay = self.current_frame['full_loop_delay_ms']
                f.write(f"闭环延迟（端到端）: {full_loop_delay:.3f} ms (从VR手部运动到WebSocket发送完成，包括网络传输)\n")
            
            # 显示延迟验证信息
            if 'total_sub_delays_ms' in self.current_frame:
                total_sub = self.current_frame['total_sub_delays_ms']
                unmeasured = self.current_frame.get('unmeasured_time_ms', 0)
                coverage = (total_sub / self.current_frame['total_delay_ms'] * 100) if self.current_frame['total_delay_ms'] > 0 else 0
                f.write(f"子延迟总和: {total_sub:.3f} ms, 未测量时间: {unmeasured:.3f} ms, 覆盖率: {coverage:.1f}%\n")
                if abs(unmeasured) > self.current_frame['total_delay_ms'] * 0.05:
                    f.write(f"⚠️  警告: 未测量时间超过总延迟的5%，可能存在遗漏或重叠的延迟记录\n")
            f.write("\n")
            
            # 添加实际帧间隔信息
            if 'actual_frame_interval_ms' in self.current_frame:
                f.write(f"实际帧间隔: {self.current_frame['actual_frame_interval_ms']:.3f} ms (包含sleep: {self.current_frame.get('sleep_time_ms', 0):.3f} ms)\n")
                f.write(f"实际帧率: {1000.0 / self.current_frame['actual_frame_interval_ms']:.2f} FPS\n")
            
            f.write("\n计算延迟:\n")
            for name, delay in self.current_frame['compute_delays'].items():
                f.write(f"  {name}: {delay:.3f} ms\n")
            
            f.write("\n传输延迟（包括通信延迟）:\n")
            # 分类显示：通信延迟和其他传输延迟
            comm_delays = {}
            other_delays = {}
            for name, delay in sorted(self.current_frame['transmit_delays'].items()):
                if name.startswith('comm_'):
                    comm_delays[name] = delay
                else:
                    other_delays[name] = delay
            
            if comm_delays:
                f.write("  通信延迟:\n")
                for name, delay in sorted(comm_delays.items()):
                    f.write(f"    {name}: {delay:.3f} ms\n")
            
            if other_delays:
                if comm_delays:
                    f.write("  其他传输延迟:\n")
                for name, delay in sorted(other_delays.items()):
                    f.write(f"    {name}: {delay:.3f} ms\n")
            
            if not comm_delays and not other_delays:
                f.write("  (无传输延迟记录)\n")
            
            f.write("\n带宽不足导致的延迟:\n")
            if self.current_frame['bandwidth_delays']:
                for name, delay in sorted(self.current_frame['bandwidth_delays'].items()):
                    f.write(f"  {name}: {delay:.3f} ms")
                    info_key = f'{name}_info'
                    if info_key in self.current_frame:
                        f.write(f" ({self.current_frame[info_key]})")
                    f.write("\n")
            else:
                f.write("  (无带宽延迟记录)\n")
            
            # 显示闭环延迟（端到端延迟）
            if 'full_loop_delay_ms' in self.current_frame:
                full_loop_delay = self.current_frame['full_loop_delay_ms']
                is_invalid = self.current_frame.get('full_loop_delay_invalid', False)
                
                if is_invalid:
                    hand_move_age = self.current_frame.get('hand_move_age_ms', 0)
                    f.write(f"\n⚠️  闭环延迟（端到端）: {full_loop_delay:.3f} ms [无效 - HAND_MOVE时间戳过旧]\n")
                    f.write(f"  起点: HAND_MOVE事件接收时间（VR设备捕获手部运动）\n")
                    f.write(f"  终点: WebSocket发送完成时间（图像发送到VR，包括网络传输）\n")
                    f.write(f"  ⚠️  警告: HAND_MOVE时间戳已过时 {hand_move_age:.3f} ms，说明VR端事件发送频率低\n")
                    f.write(f"  注意: 真正的闭环延迟应该 ≈ 主机处理延迟(30ms) + 网络传输延迟(10-50ms) + VR渲染延迟(10-20ms) ≈ 50-100ms\n")
                    f.write(f"  ⚠️  重要: 如果实际感受到的延迟接近 {full_loop_delay:.0f}ms，说明延迟可能来自：\n")
                    f.write(f"     1. 图像在共享内存中停留时间过长（检查VR端读取频率）\n")
                    f.write(f"     2. WebSocket队列积压（检查带宽和传输速度）\n")
                    f.write(f"     3. VR端帧跳过（检查VR端frame_id处理逻辑）\n")
                    if 'hand_move_receive_time' in self.current_frame:
                        hand_time = self.current_frame['hand_move_receive_time']
                        f.write(f"  时间戳: HAND_MOVE接收={hand_time:.6f} (已过时 {hand_move_age:.3f} ms)\n")
                else:
                    f.write(f"\n闭环延迟（端到端）: {full_loop_delay:.3f} ms\n")
                    f.write(f"  起点: HAND_MOVE事件接收时间（VR设备捕获手部运动）\n")
                    f.write(f"  终点: WebSocket发送完成时间（图像发送到VR，包括网络传输）\n")
                    f.write(f"  注意: 真正的显示延迟还包括VR端渲染时间（通常+10-20ms）\n")
                    if 'hand_move_receive_time' in self.current_frame and 'websocket_send_complete_time' in self.current_frame:
                        hand_time = self.current_frame['hand_move_receive_time']
                        ws_time = self.current_frame['websocket_send_complete_time']
                        f.write(f"  时间戳: HAND_MOVE接收={hand_time:.6f}, WebSocket完成={ws_time:.6f}\n")
                    
                    # 如果闭环延迟异常高（>500ms），提供诊断建议
                    if full_loop_delay > 500:
                        f.write(f"\n  ⚠️  警告: 闭环延迟异常高 ({full_loop_delay:.0f}ms)，可能原因：\n")
                        f.write(f"     1. 图像在共享内存中停留时间过长（检查VR端读取频率和帧跳过逻辑）\n")
                        f.write(f"     2. WebSocket传输队列积压（检查带宽和传输速度）\n")
                        f.write(f"     3. VR端处理速度慢（检查VR端图像处理性能）\n")
                        f.write(f"     4. 网络延迟高（检查网络连接质量）\n")
            
            f.write("\n")
    
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
            all_bandwidth_keys = set()
            for frame in self.frame_timings:
                all_compute_keys.update(frame['compute_delays'].keys())
                all_transmit_keys.update(frame['transmit_delays'].keys())
                all_bandwidth_keys.update(frame.get('bandwidth_delays', {}).keys())
            
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
            
            f.write("\n=== 带宽不足导致的延迟统计 ===\n")
            bandwidth_stats = {}
            for key in all_bandwidth_keys:
                values = [f.get('bandwidth_delays', {}).get(key, 0) for f in self.frame_timings if key in f.get('bandwidth_delays', {})]
                if values:
                    bandwidth_stats[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'total': np.sum(values)
                    }
                    f.write(f"{key}:\n")
                    f.write(f"  平均值: {bandwidth_stats[key]['mean']:.3f} ms\n")
                    f.write(f"  标准差: {bandwidth_stats[key]['std']:.3f} ms\n")
                    f.write(f"  最小值: {bandwidth_stats[key]['min']:.3f} ms\n")
                    f.write(f"  最大值: {bandwidth_stats[key]['max']:.3f} ms\n")
                    f.write(f"  累计: {bandwidth_stats[key]['total']:.3f} ms\n\n")
            
            # 总延迟统计
            total_delays = [f['total_delay_ms'] for f in self.frame_timings]
            fps_list = [f['fps'] for f in self.frame_timings]
            
            # 闭环延迟统计（如果存在）
            full_loop_delays = [f.get('full_loop_delay_ms') for f in self.frame_timings if 'full_loop_delay_ms' in f and not f.get('full_loop_delay_invalid', False)]
            invalid_count = sum(1 for f in self.frame_timings if f.get('full_loop_delay_invalid', False))
            
            if full_loop_delays:
                f.write("\n=== 闭环延迟统计（端到端）===\n")
                f.write(f"有效闭环延迟帧数: {len(full_loop_delays)}\n")
                if invalid_count > 0:
                    f.write(f"⚠️  无效闭环延迟帧数: {invalid_count} (HAND_MOVE时间戳过旧)\n")
                f.write(f"闭环延迟平均值: {np.mean(full_loop_delays):.3f} ms\n")
                f.write(f"闭环延迟标准差: {np.std(full_loop_delays):.3f} ms\n")
                f.write(f"闭环延迟最小值: {np.min(full_loop_delays):.3f} ms\n")
                f.write(f"闭环延迟最大值: {np.max(full_loop_delays):.3f} ms\n")
                f.write(f"注意: 真正的显示延迟还包括VR端渲染时间（通常+10-20ms）\n")
                f.write(f"说明: 闭环延迟 = VR设备捕获手部运动 -> 主机处理 -> 图像发送到VR（包括网络传输）\n")
                if invalid_count > 0:
                    f.write(f"\n⚠️  警告: {invalid_count} 帧的闭环延迟无效，因为HAND_MOVE时间戳过旧（>200ms）\n")
                    f.write(f"   这说明VR端HAND_MOVE事件发送频率可能过低，或VR端处理有问题\n")
                f.write("\n")
            elif invalid_count > 0:
                f.write("\n=== 闭环延迟统计（端到端）===\n")
                f.write(f"⚠️  所有闭环延迟都无效: {invalid_count} 帧\n")
                f.write(f"原因: HAND_MOVE时间戳过旧（>200ms），说明VR端事件发送频率过低\n")
                f.write(f"建议: 检查VR端HAND_MOVE事件发送频率，确保每帧都有新的事件\n\n")
            
            # 实际帧间隔统计（如果存在）
            actual_intervals = [f.get('actual_frame_interval_ms', f['total_delay_ms']) for f in self.frame_timings]
            sleep_times = [f.get('sleep_time_ms', 0) for f in self.frame_timings]
            
            f.write("\n=== 总体延迟统计 ===\n")
            f.write(f"总延迟平均值: {np.mean(total_delays):.3f} ms\n")
            f.write(f"总延迟标准差: {np.std(total_delays):.3f} ms\n")
            f.write(f"总延迟最小值: {np.min(total_delays):.3f} ms\n")
            f.write(f"总延迟最大值: {np.max(total_delays):.3f} ms\n\n")
            
            if any(actual_intervals):
                f.write("\n=== 实际帧间隔统计（包含sleep）===\n")
                f.write(f"实际帧间隔平均值: {np.mean(actual_intervals):.3f} ms\n")
                f.write(f"实际帧间隔标准差: {np.std(actual_intervals):.3f} ms\n")
                f.write(f"实际帧间隔最小值: {np.min(actual_intervals):.3f} ms\n")
                f.write(f"实际帧间隔最大值: {np.max(actual_intervals):.3f} ms\n")
                f.write(f"Sleep时间平均值: {np.mean(sleep_times):.3f} ms\n")
                f.write(f"Sleep时间占比: {np.mean(sleep_times) / np.mean(actual_intervals) * 100:.1f}%\n\n")
            
            f.write("\n=== 帧率统计 ===\n")
            f.write(f"理论帧率（基于总延迟）: 平均={np.mean(fps_list):.2f} FPS\n")
            f.write(f"理论帧率标准差: {np.std(fps_list):.2f} FPS\n")
            f.write(f"最低帧率: {np.min(fps_list):.2f} FPS\n")
            f.write(f"最高帧率: {np.max(fps_list):.2f} FPS\n")
            if any(actual_intervals):
                actual_fps = [1000.0 / interval for interval in actual_intervals if interval > 0]
                f.write(f"\n实际帧率（基于帧间隔）: 平均={np.mean(actual_fps):.2f} FPS\n")
                f.write(f"实际帧率标准差: {np.std(actual_fps):.2f} FPS\n")
                f.write(f"最低实际帧率: {np.min(actual_fps):.2f} FPS\n")
                f.write(f"最高实际帧率: {np.max(actual_fps):.2f} FPS\n")
            f.write("\n")
            
            # 延迟占比分析
            if compute_stats or transmit_stats or bandwidth_stats:
                f.write("\n=== 延迟占比分析 ===\n")
                total_compute_time = sum([s['total'] for s in compute_stats.values()])
                total_transmit_time = sum([s['total'] for s in transmit_stats.values()])
                total_bandwidth_time = sum([s['total'] for s in bandwidth_stats.values()])
                total_frame_time = sum(total_delays)
                
                f.write(f"总计算时间: {total_compute_time:.3f} ms ({total_compute_time/total_frame_time*100:.2f}%)\n")
                f.write(f"总传输时间: {total_transmit_time:.3f} ms ({total_transmit_time/total_frame_time*100:.2f}%)\n")
                f.write(f"总带宽延迟: {total_bandwidth_time:.3f} ms ({total_bandwidth_time/total_frame_time*100:.2f}%)\n")
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
                
                f.write("\n各带宽延迟模块占比:\n")
                for key, stats in bandwidth_stats.items():
                    percentage = stats['total'] / total_frame_time * 100
                    f.write(f"  {key}: {percentage:.2f}%\n")
            
            f.write(f"\n{'='*80}\n")
            f.write(f"=== 分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")


class VuerTeleop:
    def __init__(self, config_file_path, latency_monitor=None, sim=None):
        # 全景图分辨率：2:1的宽高比（提高画质：高度800，宽度1600）
        # 降采样后是400×1600，比2D模式的360×1280更高，提供更好的画质
        self.resolution = (800, 1600)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)
        
        # 全景图的宽高比为2:1
        self.panorama_height = self.resolution_cropped[0]  # 800
        self.panorama_width = self.panorama_height * 2     # 1600
        
        self.img_shape = (self.panorama_height, self.panorama_width, 3)
        self.img_height, self.img_width = self.panorama_height, self.panorama_width

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        # 帧元数据共享：8字节int64(frame_id) + 8字节float64(t0) + 8字节float64(hand_move_receive_time) + 8字节float64(websocket_send_complete_time)
        self.meta_shm = shared_memory.SharedMemory(create=True, size=32)  # 扩展到32字节以包含HAND_MOVE接收时间和WebSocket发送完成时间
        image_queue = Queue()
        toggle_streaming = Event()
        # 使用全景模式：传入全景图尺寸而不是原分辨率
        self.tv = OpenTeleVision(self.img_shape, self.shm.name, image_queue, toggle_streaming, ngrok=True, panorama_mode=True, meta_shm_name=self.meta_shm.name)
        self.processor = VuerPreprocessor()
        self.latency_monitor = latency_monitor
        # 可选：保存模拟器引用用于调试（例如相机位置）
        self.sim = sim

        # 使用模块级别的项目根目录
        assets_dir_str = str(_ASSETS_DIR)
        if not _ASSETS_DIR.exists():
            raise ValueError(f"Assets directory not found: {assets_dir_str}. Expected: {_ASSETS_DIR}")
        RetargetingConfig.set_default_urdf_dir(assets_dir_str)
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        # 注意：HAND_MOVE接收时间现在在main循环中从meta_shm读取
        # 从预处理器获取处理后的姿态数据
        # head_mat: (4, 4) - 头部齐次变换矩阵
        # left_wrist_mat: (4, 4) - 左手腕齐次变换矩阵（已转换为Z-UP坐标系，相对头部）
        # right_wrist_mat: (4, 4) - 右手腕齐次变换矩阵（已转换为Z-UP坐标系，相对头部）
        # left_hand_mat: (25, 3) - 左手25个关键点的3D坐标（已转换为相对手腕的Inspire Hand坐标系）
        # right_hand_mat: (25, 3) - 右手25个关键点的3D坐标（已转换为相对手腕的Inspire Hand坐标系）
        
        # 监控processor处理延迟（包含从共享内存读取数据+计算的延迟）
        # 注意：processor.process()内部会从共享内存读取数据（通过.tv.head_matrix.copy()等）
        # 由于读取发生在process()内部，我们无法在外部精确分离读取和计算时间
        # 因此，我们只记录总处理时间，不单独记录读取时间以避免重复计算
        if self.latency_monitor:
            t_process_start = time.time()
        
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
        
        if self.latency_monitor:
            t_process_end = time.time()
            # 记录总处理延迟（包括读取共享内存和计算，作为计算延迟）
            # 注意：虽然包含读取操作，但这是处理流程的一部分，归类为计算延迟更合理
            self.latency_monitor.record_compute("teleop_preprocessor", t_process_start, t_process_end)
        
        # 调试：打印头部位置（用于检查高度对齐问题）
        if not hasattr(self, '_head_pos_logged'):
            head_pos = head_mat[:3, 3]
            cam_pos = self.sim.cam_pos if self.sim and hasattr(self.sim, 'cam_pos') else np.array([-0.6, 0, 1.6])
            print(f"\n=== 头部位置调试信息 ===")
            print(f"头部世界位置 (Isaac Gym坐标系): {head_pos}")
            print(f"相机位置: {cam_pos}")
            print(f"手部偏移量(=相机位置): {cam_pos}")
            print(f"高度差异 (头部Z - 相机Z): {head_pos[2] - cam_pos[2]:.3f} 米")
            print(f"位置差异 (头部 - 相机位置): {head_pos - cam_pos}")
            print(f"\n手部位置计算（与2D模式完全一致）:")
            print(f"  - left_wrist_mat[:3, 3] 是相对头部的向量（世界坐标系）")
            print(f"  - offset = 相机位置 cam_pos = [-0.6, 0, 1.6]")
            print(f"  - 最终手部位置 = 相对位置 + offset")
            print(f"\n注意：")
            print(f"  - 在2D模式下，相机跟随头部，所以手部和相机都相对于头部位置")
            print(f"  - 在Panorama模式下，相机固定在 {cam_pos}，但手部位置计算与2D完全一致")
            if np.linalg.norm(head_pos - cam_pos) < 0.1:
                print(f"\n✓ 相机位置已设置为头部初始位置，确保VR视角与头部位置匹配")
            else:
                print(f"\n注意：头部位置可能已移动，相机位置保持在初始位置（panorama模式特性）")
            self._head_pos_logged = True

        # 提取头部旋转矩阵（3×3）
        # 注意：在panorama模式下，手部位置计算完全模仿2D模式，不使用head_rmat转换
        if self.latency_monitor:
            t_extract_start = time.time()
        
        head_rmat = head_mat[:3, :3]  # shape: (3, 3) - 头部旋转矩阵（保留用于其他用途，手部位置计算不使用）
        
        if self.latency_monitor:
            t_extract_end = time.time()
            self.latency_monitor.record_compute("teleop_extract_head_rmat", t_extract_start, t_extract_end)

        # 构建左右手位姿（计算延迟）
        # 完全模仿2D模式的实现逻辑：使用手部相对于头部的位置，不应用任何旋转转换
        # 2D模式：left_pose = left_wrist_mat[:3, 3] + offset
        # Panorama模式：完全相同的实现，即使相机不旋转也不影响使用相对位置
        if self.latency_monitor:
            t_left_pose_start = time.time()
        
        # 与2D模式完全一致：使用相对位置 + 固定偏移量（相机位置）
        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        
        if self.latency_monitor:
            t_left_pose_end = time.time()
            self.latency_monitor.record_compute("teleop_build_left_pose", t_left_pose_start, t_left_pose_end)

        if self.latency_monitor:
            t_right_pose_start = time.time()
        
        # 与2D模式完全一致：使用相对位置 + 固定偏移量（相机位置）
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        
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

        # 使用模块级别的项目根目录
        asset_root = str(_ASSETS_DIR)
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
        pose.p = gymapi.Vec3(0, 0.3, 1.25)
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

        # 相机位置：固定在Isaac Gym坐标系的(-0.7, 0, 1.75)
        # 该位置作为全景拍摄的基准点
        self.cam_pos = np.array([-0.75, 0, 1.7])

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
        # VuerTeleop使用800高度，所以每面也是800（提高画质）
        self.face_resolution = 800
        self.panorama_height = self.face_resolution  # 800
        self.panorama_width = self.panorama_height * 2  # 1600
        
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
            #if face_name in ['R', 'B']:
            #    img = np.fliplr(img)
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
            
            # 水平翻转全景图以修正左右相反的问题
            panorama_uint8 = np.fliplr(panorama_uint8)
            
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
    parser.add_argument('--latency-log', type=str, default=None, help='Path to latency log file (auto-generated if not specified)')
    parser.add_argument('--network', action='store_true', help='Network mode (for cross-network transmission)')
    args = parser.parse_args()
    
    # 自动生成日志文件名
    if args.latency_log is None:
        args.latency_log = 'latency_log_panorama_network.txt' if args.network else 'latency_log_panorama_local.txt'
    
    # 创建延迟监控器
    latency_monitor = LatencyMonitor(log_file_path=args.latency_log, network_mode=args.network)
    
    # 获取配置文件路径（相对于脚本所在目录）
    config_path = _SCRIPT_DIR / 'inspire_hand.yml'
    simulator = Sim(print_freq=args.print_freq, use_panorama=not args.no_panorama, latency_monitor=latency_monitor)
    teleoperator = VuerTeleop(str(config_path), latency_monitor=latency_monitor, sim=simulator)

    frame_count = 0
    try:
        while True:
            # 记录循环开始时间（在start_frame之前，用于计算实际帧间隔）
            loop_start = time.time()
            # 开始新的一帧
            if latency_monitor:
                latency_monitor.start_frame()
                # 覆盖start_time为loop_start，确保总延迟计算准确（从循环真正开始计算）
                latency_monitor.current_frame['start_time'] = loop_start
                latency_monitor.current_frame['timestamps']['frame_start'] = loop_start
                latency_monitor.current_frame['timestamps']['loop_start'] = loop_start
            
            # Teleoperator步骤
            # 帧ID与起始时间（端到端测准）
            if not hasattr(teleoperator, '_frame_id'):
                teleoperator._frame_id = 0
            else:
                teleoperator._frame_id += 1
            # 写入meta共享内存：frame_id(int64)与t0(float64)
            # 注意：hand_move_receive_time和websocket_send_complete_time由TeleVision_panorama.py写入
            try:
                mv = memoryview(teleoperator.meta_shm.buf)
                
                # 在写入新的frame_id之前，先读取当前HAND_MOVE接收时间（对应当前帧）
                # 这是处理当前帧时，最近一次HAND_MOVE事件的接收时间
                current_hand_move_time = float(np.frombuffer(mv[16:24], dtype=np.float64, count=1)[0])
                
                # 写入新的frame_id和t0（使用numpy数组方式正确写入共享内存）
                frame_id_arr = np.ndarray(1, dtype=np.int64, buffer=mv[:8])
                frame_id_arr[0] = teleoperator._frame_id
                t0_arr = np.ndarray(1, dtype=np.float64, buffer=mv[8:16])
                t0_arr[0] = time.time()
                
                # 如果HAND_MOVE接收时间有效，记录为当前帧的闭环延迟起点
                if current_hand_move_time > 0 and latency_monitor:
                    latency_monitor.current_frame['hand_move_receive_time'] = current_hand_move_time
                    latency_monitor.current_frame['timestamps']['hand_move_receive'] = current_hand_move_time
                    
                    # 读取上一帧的WebSocket发送完成时间（用于计算上一帧的完整闭环延迟）
                    prev_websocket_complete_time = float(np.frombuffer(mv[24:32], dtype=np.float64, count=1)[0])
                    if prev_websocket_complete_time > 0:
                        # 计算上一帧的完整闭环延迟：从HAND_MOVE接收到WebSocket发送完成
                        # 注意：这个延迟会在上一帧的延迟日志中显示
                        # 这里我们保存到当前帧，用于统计
                        if hasattr(latency_monitor, '_prev_frame_loop_delay'):
                            # 将上一帧的闭环延迟保存到上一帧数据中（如果可能）
                            pass
            except Exception:
                pass
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            
            # Simulator步骤
            panorama_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
            
            # 图像传输延迟（写入共享内存）
            if latency_monitor:
                t_img_transmit_start = time.time()
            
            np.copyto(teleoperator.img_array, panorama_img)
            
            if latency_monitor:
                t_img_transmit_end = time.time()
                latency_monitor.record_transmit("image_shared_memory_write", t_img_transmit_start, t_img_transmit_end)
                
                # 记录图像发送完成时间（主机端处理完成）
                latency_monitor.current_frame['image_send_complete_time'] = t_img_transmit_end
                latency_monitor.current_frame['timestamps']['image_send_complete'] = t_img_transmit_end
                
                # 注意：完整的闭环延迟（从HAND_MOVE接收到WebSocket发送完成）会在下一帧计算
                # 因为WebSocket发送完成时间由TeleVision_panorama.py在异步函数中写入
            
            # 记录实际帧处理完成时间（不包括sleep）
            loop_elapsed = time.time() - loop_start
            frame_end_before_sleep = loop_start + loop_elapsed  # 帧处理完成时间（不包括sleep）
            
            # 保持33fps：目标帧间隔为0.03秒（1000/33.33 ≈ 30ms）
            # 如果处理时间小于30ms，则sleep剩余时间以保持稳定的帧率
            sleep_remain = 0.03 - loop_elapsed
            sleep_time = 0.0
            if sleep_remain > 0:
                sleep_start = time.time()
                time.sleep(sleep_remain)
                sleep_time = time.time() - sleep_start
                if latency_monitor:
                    # 记录sleep时间（用于分析实际帧率）
                    latency_monitor.current_frame['sleep_time'] = sleep_time * 1000
                    latency_monitor.current_frame['actual_frame_interval'] = (loop_elapsed + sleep_time) * 1000
            
            # 结束当前帧，写入延迟信息
            # 注意：总延迟不包括sleep时间，因为sleep是人为添加的延迟
            if latency_monitor:
                # 记录实际帧间隔（包括sleep）
                latency_monitor.current_frame['timestamps']['frame_end_before_sleep'] = frame_end_before_sleep
                latency_monitor.current_frame['actual_frame_interval_ms'] = (loop_elapsed + sleep_time) * 1000
                latency_monitor.current_frame['sleep_time_ms'] = sleep_time * 1000
                latency_monitor.current_frame['_end_time_with_sleep'] = time.time()  # 保存包含sleep的结束时间用于分析
                
                # 尝试读取当前帧对应的WebSocket发送完成时间（如果已经完成）
                # 这用于计算完整的闭环延迟
                try:
                    mv = memoryview(teleoperator.meta_shm.buf)
                    websocket_complete_time = float(np.frombuffer(mv[24:32], dtype=np.float64, count=1)[0])
                    if 'hand_move_receive_time' in latency_monitor.current_frame and websocket_complete_time > 0:
                        hand_move_time = latency_monitor.current_frame['hand_move_receive_time']
                        # 只有当前帧的WebSocket完成时间才计算（通过时间戳比较）
                        if websocket_complete_time >= hand_move_time:
                            full_loop_delay = (websocket_complete_time - hand_move_time) * 1000  # 转换为毫秒
                            
                            # 验证：如果HAND_MOVE时间戳过旧（超过200ms），说明VR端事件发送频率低
                            # 这种情况下，闭环延迟不准确，应该标记为无效
                            time_since_hand_move = (frame_end_before_sleep - hand_move_time) * 1000  # 毫秒
                            if time_since_hand_move > 200:
                                # HAND_MOVE时间戳过旧，标记为无效，但记录实际值用于诊断
                                latency_monitor.current_frame['full_loop_delay_ms'] = full_loop_delay
                                latency_monitor.current_frame['full_loop_delay_invalid'] = True
                                latency_monitor.current_frame['hand_move_age_ms'] = time_since_hand_move
                            else:
                                # 正常情况：HAND_MOVE时间戳是新鲜的
                                latency_monitor.current_frame['full_loop_delay_ms'] = full_loop_delay
                                latency_monitor.current_frame['full_loop_delay_invalid'] = False
                                latency_monitor.current_frame['websocket_send_complete_time'] = websocket_complete_time
                                latency_monitor.current_frame['timestamps']['websocket_send_complete'] = websocket_complete_time
                except Exception:
                    pass
                
                # 使用frame_end_before_sleep作为总延迟计算的结束时间（不包括sleep）
                latency_monitor.end_frame(end_time=frame_end_before_sleep)
            
            frame_count += 1
    except KeyboardInterrupt:
        # 生成延迟统计摘要
        if latency_monitor:
            latency_monitor.generate_summary()
        simulator.end()
        exit(0)


