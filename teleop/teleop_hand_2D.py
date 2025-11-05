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
from datetime import datetime


class LatencyMonitor:
    """延迟监控类，用于追踪2D pipeline的计算延迟、传输延迟和带宽不足导致的延迟"""
    def __init__(self, log_file_path="latency_log_2d_local.txt", network_mode=False):
        self.log_file_path = log_file_path
        self.frame_timings = []
        self.current_frame = {}
        self.frame_count = 0
        self.network_mode = network_mode
        
        # 创建日志文件并写入头部
        mode_str = "网络模式" if network_mode else "本地模式"
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 2D Pipeline延迟监控日志 ({mode_str}) - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
            f.write("监控内容：\n")
            f.write("  - 计算延迟：各模块的计算时间\n")
            f.write("  - 传输延迟：数据读写、共享内存、网络传输等\n")
            f.write("  - 带宽延迟：带宽不足导致的队列阻塞、传输延迟等\n\n")
    
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
    
    def end_frame(self):
        """结束当前帧，计算总延迟和帧率，写入文件"""
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
                if self.network_mode:
                    f.write(f"闭环延迟（端到端）: {full_loop_delay:.3f} ms (从VR手部运动到WebSocket发送完成，⚠️ 注意：不包含主机->VR的网络传输时间)\n")
                    f.write(f"  ⚠️  重要：session.upsert是异步的，只记录主机端将数据放入队列的时间，实际的网络传输（主机->VR）没有被测量\n")
                    if 'complete_full_loop_delay_ms' in self.current_frame:
                        complete_delay = self.current_frame.get('complete_full_loop_delay_ms', full_loop_delay)
                        f.write(f"  完整的闭环延迟（包含所有5个阶段）: {complete_delay:.3f} ms\n")
                else:
                    f.write(f"闭环延迟（端到端）: {full_loop_delay:.3f} ms (从VR手部运动到WebSocket发送完成，本地模式)\n")
            
            # 显示延迟验证信息
            if 'total_sub_delays_ms' in self.current_frame:
                total_sub = self.current_frame['total_sub_delays_ms']
                unmeasured = self.current_frame.get('unmeasured_time_ms', 0)
                coverage = (total_sub / self.current_frame['total_delay_ms'] * 100) if self.current_frame['total_delay_ms'] > 0 else 0
                f.write(f"子延迟总和: {total_sub:.3f} ms, 未测量时间: {unmeasured:.3f} ms, 覆盖率: {coverage:.1f}%\n")
                if abs(unmeasured) > self.current_frame['total_delay_ms'] * 0.05:
                    f.write(f"⚠️  警告: 未测量时间超过总延迟的5%，可能存在遗漏或重叠的延迟记录\n")
            f.write("\n")
            
            f.write("计算延迟:\n")
            if self.current_frame['compute_delays']:
                for name, delay in sorted(self.current_frame['compute_delays'].items()):
                    f.write(f"  {name}: {delay:.3f} ms\n")
            else:
                f.write("  (无计算延迟记录)\n")
            
            f.write("\n传输延迟:\n")
            if self.current_frame['transmit_delays']:
                for name, delay in sorted(self.current_frame['transmit_delays'].items()):
                    f.write(f"  {name}: {delay:.3f} ms\n")
            else:
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
                    if 'hand_move_receive_time' in self.current_frame:
                        hand_time = self.current_frame['hand_move_receive_time']
                        f.write(f"  时间戳: HAND_MOVE接收={hand_time:.6f} (已过时 {hand_move_age:.3f} ms)\n")
                else:
                    if self.network_mode:
                        f.write(f"\n⚠️  闭环延迟（端到端）: {full_loop_delay:.3f} ms [不包含网络传输时间]\n")
                        f.write(f"  起点: HAND_MOVE事件接收时间（VR设备捕获手部运动）\n")
                        f.write(f"  终点: WebSocket发送完成时间（主机端将数据放入发送队列，⚠️ 不包含实际网络传输）\n")
                        f.write(f"  ⚠️  关键问题：session.upsert是异步的，只记录主机端将数据放入队列的时间，不包含主机->VR的网络传输\n")
                        if 'complete_full_loop_delay_ms' in self.current_frame:
                            measured_delay = self.current_frame.get('measured_delay_ms', full_loop_delay)
                            estimated_vr_to_host = self.current_frame.get('estimated_vr_to_host_transmission_ms', 0)
                            host_processing = self.current_frame.get('host_processing_time_ms', 0)
                            vr_processing = self.current_frame.get('vr_processing_time_ms', 0)
                            estimated_host_to_vr_transmission = self.current_frame.get('estimated_host_to_vr_transmission_ms', 0)
                            estimated_vr_rendering = self.current_frame.get('estimated_vr_rendering_ms', 0)
                            complete_delay = self.current_frame.get('complete_full_loop_delay_ms', full_loop_delay)
                            
                            queue_blocking = self.current_frame.get('queue_blocking_delay_ms', 0)
                            base_transmission = self.current_frame.get('estimated_host_to_vr_transmission_base_ms', estimated_host_to_vr_transmission)
                            
                            f.write(f"  完整闭环延迟分解（VR设备捕获 → VR设备显示）:\n")
                            f.write(f"    1. VR→主机传输: {estimated_vr_to_host:.3f} ms\n")
                            f.write(f"    2. 主机处理: {host_processing:.3f} ms\n")
                            f.write(f"    3. VR处理（到WebSocket发送）: {vr_processing:.3f} ms\n")
                            f.write(f"    4. 主机→VR传输:\n")
                            f.write(f"       - 基础传输延迟: {base_transmission:.3f} ms（估算）\n")
                            if queue_blocking > 0:
                                f.write(f"       - 队列阻塞延迟: {queue_blocking:.3f} ms\n")
                            else:
                                f.write(f"       - 队列阻塞延迟: 0.000 ms（无阻塞）\n")
                            f.write(f"       - 总传输延迟: {estimated_host_to_vr_transmission:.3f} ms\n")
                            f.write(f"    5. VR渲染: {estimated_vr_rendering:.3f} ms（估算）\n")
                            f.write(f"    测量的延迟（阶段1-3）: {measured_delay:.3f} ms\n")
                            f.write(f"    完整的闭环延迟（阶段1-5）: {complete_delay:.3f} ms\n")
                            f.write(f"  说明: 使用ngrok中转（东京），典型主机->VR传输延迟: 50-100ms，VR渲染: 10-20ms\n")
                        else:
                            f.write(f"  ⚠️  警告: 无法估算网络延迟，实际延迟可能远大于显示值\n")
                        if 'hand_move_receive_time' in self.current_frame and 'websocket_send_complete_time' in self.current_frame:
                            hand_time = self.current_frame['hand_move_receive_time']
                            ws_time = self.current_frame['websocket_send_complete_time']
                            f.write(f"  时间戳: HAND_MOVE接收={hand_time:.6f}, WebSocket完成={ws_time:.6f}\n")
                    else:
                        f.write(f"\n闭环延迟（端到端）: {full_loop_delay:.3f} ms\n")
                        f.write(f"  起点: HAND_MOVE事件接收时间（VR设备捕获手部运动）\n")
                        f.write(f"  终点: WebSocket发送完成时间（图像发送到VR，本地模式）\n")
                        # 在本地模式下，同样输出分解（主机→VR传输与VR渲染近似0）
                        if 'complete_full_loop_delay_ms' in self.current_frame:
                            measured_delay = self.current_frame.get('measured_delay_ms', full_loop_delay)
                            estimated_vr_to_host = self.current_frame.get('estimated_vr_to_host_transmission_ms', 0)
                            host_processing = self.current_frame.get('host_processing_time_ms', 0)
                            vr_processing = self.current_frame.get('vr_processing_time_ms', 0)
                            estimated_host_to_vr_transmission = self.current_frame.get('estimated_host_to_vr_transmission_ms', 0)
                            estimated_vr_rendering = self.current_frame.get('estimated_vr_rendering_ms', 0)
                            complete_delay = self.current_frame.get('complete_full_loop_delay_ms', full_loop_delay)

                            f.write(f"  完整闭环延迟分解（VR设备捕获 → VR设备显示，本地）:\n")
                            f.write(f"    1. VR→主机传输: {estimated_vr_to_host:.3f} ms\n")
                            f.write(f"    2. 主机处理: {host_processing:.3f} ms\n")
                            f.write(f"    3. VR处理（到WebSocket发送）: {vr_processing:.3f} ms\n")
                            f.write(f"    4. 主机→VR传输: {estimated_host_to_vr_transmission:.3f} ms（本地≈0）\n")
                            f.write(f"    5. VR渲染: {estimated_vr_rendering:.3f} ms（本地≈0）\n")
                            f.write(f"    测量的延迟（阶段1-3）: {measured_delay:.3f} ms\n")
                            f.write(f"    完整的闭环延迟（阶段1-5）: {complete_delay:.3f} ms\n")
                        if 'hand_move_receive_time' in self.current_frame and 'websocket_send_complete_time' in self.current_frame:
                            hand_time = self.current_frame['hand_move_receive_time']
                            ws_time = self.current_frame['websocket_send_complete_time']
                            f.write(f"  时间戳: HAND_MOVE接收={hand_time:.6f}, WebSocket完成={ws_time:.6f}\n")
            
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
                if self.network_mode:
                    f.write(f"\n⚠️  重要：网络模式下的闭环延迟统计\n")
                    f.write(f"  显示的值不包含主机->VR的网络传输时间（单程）\n")
                    f.write(f"  原因：session.upsert是异步的，只记录主机端将数据放入队列的时间\n")
                    f.write(f"  注意：hand_move_receive_time已包含VR->主机的网络传输\n")
                    # 计算完整的闭环延迟统计（包含所有5个阶段）
                    complete_delays = [f.get('complete_full_loop_delay_ms', 0) for f in self.frame_timings if 'complete_full_loop_delay_ms' in f and f.get('complete_full_loop_delay_ms', 0) > 0]
                    measured_delays = [f.get('measured_delay_ms', 0) for f in self.frame_timings if 'measured_delay_ms' in f]
                    estimated_vr_to_hosts = [f.get('estimated_vr_to_host_transmission_ms', 0) for f in self.frame_timings if 'estimated_vr_to_host_transmission_ms' in f]
                    estimated_host_to_vr_transmissions = [f.get('estimated_host_to_vr_transmission_ms', 0) for f in self.frame_timings if 'estimated_host_to_vr_transmission_ms' in f]
                    estimated_vr_renderings = [f.get('estimated_vr_rendering_ms', 0) for f in self.frame_timings if 'estimated_vr_rendering_ms' in f]
                    
                    if complete_delays:
                        f.write(f"  完整的闭环延迟统计（阶段1-5）:\n")
                        f.write(f"    平均值: {np.mean(complete_delays):.3f} ms\n")
                        f.write(f"    最小值: {np.min(complete_delays):.3f} ms\n")
                        f.write(f"    最大值: {np.max(complete_delays):.3f} ms\n")
                        f.write(f"    标准差: {np.std(complete_delays):.3f} ms\n")
                    
                    if measured_delays:
                        f.write(f"  测量的延迟统计（阶段1-3，不含主机→VR传输和VR渲染）:\n")
                        f.write(f"    平均值: {np.mean(measured_delays):.3f} ms\n")
                    
                    if estimated_vr_to_hosts:
                        avg_vr_to_host = np.mean([d for d in estimated_vr_to_hosts if d > 0])
                        f.write(f"  平均VR→主机传输延迟（阶段1）: {avg_vr_to_host:.3f} ms\n")
                    
                    if estimated_host_to_vr_transmissions:
                        avg_host_to_vr = np.mean([d for d in estimated_host_to_vr_transmissions if d > 0])
                        # 计算平均队列阻塞延迟
                        queue_blocking_delays = [f.get('queue_blocking_delay_ms', 0) for f in self.frame_timings if 'queue_blocking_delay_ms' in f]
                        avg_queue_blocking = np.mean([d for d in queue_blocking_delays if d > 0]) if queue_blocking_delays else 0
                        base_transmissions = [f.get('estimated_host_to_vr_transmission_base_ms', 70.0) for f in self.frame_timings if 'estimated_host_to_vr_transmission_base_ms' in f]
                        avg_base_transmission = np.mean([d for d in base_transmissions if d > 0]) if base_transmissions else 70.0
                        
                        f.write(f"  平均主机→VR传输延迟（阶段4）:\n")
                        f.write(f"    基础传输延迟: {avg_base_transmission:.3f} ms（估算）\n")
                        if avg_queue_blocking > 0:
                            f.write(f"    队列阻塞延迟: {avg_queue_blocking:.3f} ms\n")
                        f.write(f"    总传输延迟: {avg_host_to_vr:.3f} ms\n")
                    
                    if estimated_vr_renderings:
                        avg_vr_rendering = np.mean([d for d in estimated_vr_renderings if d > 0])
                        f.write(f"  平均VR渲染延迟（阶段5，估算）: {avg_vr_rendering:.3f} ms\n")
                    f.write(f"  说明: 使用ngrok中转（东京），典型主机->VR传输延迟: 50-100ms，加上VR渲染: 10-20ms\n")
                else:
                    f.write(f"注意: 真正的显示延迟还包括VR端渲染时间（通常+10-20ms）\n")
                f.write(f"说明: 闭环延迟 = VR设备捕获手部运动 -> 主机处理 -> 图像发送到VR（{'不包含' if self.network_mode else '包括'}网络传输）\n")
                if invalid_count > 0:
                    f.write(f"\n⚠️  警告: {invalid_count} 帧的闭环延迟无效，因为HAND_MOVE时间戳过旧（>200ms）\n")
                    f.write(f"   这说明VR端HAND_MOVE事件发送频率可能过低，或VR端处理有问题\n")
                f.write("\n")
            elif invalid_count > 0:
                f.write("\n=== 闭环延迟统计（端到端）===\n")
                f.write(f"⚠️  所有闭环延迟都无效: {invalid_count} 帧\n")
                f.write(f"原因: HAND_MOVE时间戳过旧（>200ms），说明VR端事件发送频率过低\n")
                f.write(f"建议: 检查VR端HAND_MOVE事件发送频率，确保每帧都有新的事件\n\n")
            
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
    def __init__(self, config_file_path, latency_monitor=None):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        # 帧元数据共享：8字节int64(frame_id) + 8字节float64(t0) + 8字节float64(hand_move_receive_time) + 8字节float64(websocket_send_complete_time)
        self.meta_shm = shared_memory.SharedMemory(create=True, size=32)  # 扩展到32字节以包含HAND_MOVE接收时间和WebSocket发送完成时间
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=True, meta_shm_name=self.meta_shm.name)
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
        # 处理VR输入数据（计算延迟）
        # 注意：processor.process()内部会从共享内存读取数据（通过.tv.head_matrix.copy()等）
        # 由于读取发生在process()内部，我们无法在外部精确分离读取和计算时间
        # 因此，我们只记录总处理时间，不单独记录读取时间以避免重复计算
        if self.latency_monitor:
            t_process_start = time.time()
        
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
        
        if self.latency_monitor:
            t_process_end = time.time()
            # 记录总处理延迟（包括读取共享内存和计算，作为计算延迟）
            self.latency_monitor.record_compute("teleop_preprocessor", t_process_start, t_process_end)
        
        # 提取头部旋转矩阵（计算延迟）
        # 确保时间戳连续：使用上一个操作的结束时间作为开始时间
        if self.latency_monitor:
            t_extract_start = t_process_end  # 连续，无间隙
        
        head_rmat = head_mat[:3, :3]
        
        if self.latency_monitor:
            t_extract_end = time.time()
            self.latency_monitor.record_compute("teleop_extract_head_rmat", t_extract_start, t_extract_end)
        
        # 构建左右手位姿（计算延迟）
        # 确保时间戳连续
        if self.latency_monitor:
            t_left_pose_start = t_extract_end  # 连续，无间隙
        
        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        
        if self.latency_monitor:
            t_left_pose_end = time.time()
            self.latency_monitor.record_compute("teleop_build_left_pose", t_left_pose_start, t_left_pose_end)
        
        # 确保时间戳连续
        if self.latency_monitor:
            t_right_pose_start = t_left_pose_end  # 连续，无间隙
        
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        
        if self.latency_monitor:
            t_right_pose_end = time.time()
            self.latency_monitor.record_compute("teleop_build_right_pose", t_right_pose_start, t_right_pose_end)
        
        # 手部重定向（计算延迟）
        # 确保时间戳连续
        if self.latency_monitor:
            t_retarget_start = t_right_pose_end  # 连续，无间隙
        
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        
        if self.latency_monitor:
            t_retarget_end = time.time()
            self.latency_monitor.record_compute("teleop_hand_retargeting", t_retarget_start, t_retarget_end)

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos

class Sim:
    def __init__(self,
                 print_freq=False,
                 latency_monitor=None):
        self.print_freq = print_freq
        self.latency_monitor = latency_monitor

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

        # 获取assets目录的绝对路径（与panorama版本保持一致）
        asset_root = str(Path(__file__).parent.parent / 'assets')
        left_asset_path = "inspire_hand/inspire_hand_left.urdf"
        right_asset_path = "inspire_hand/inspire_hand_right.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)  # 显式转换为int避免DeprecationWarning
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        
        if left_asset is None or right_asset is None:
            print(f"*** Failed to load hand assets from: {asset_root}")
            print(f"    Left asset path: {left_asset_path}")
            print(f"    Right asset path: {right_asset_path}")
            quit()
        
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
        # 设置actor根状态（传输延迟）
        if self.latency_monitor:
            t_set_state_start = time.time()
        
        self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        
        if self.latency_monitor:
            t_set_state_end = time.time()
            self.latency_monitor.record_transmit("sim_set_actor_root_state", t_set_state_start, t_set_state_end)
        
        # 设置actor DOF状态（传输延迟）
        # 确保时间戳连续
        if self.latency_monitor:
            t_set_dof_start = t_set_state_end  # 连续，无间隙
        
        left_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        left_states['pos'] = left_qpos
        self.gym.set_actor_dof_states(self.env, self.left_handle, left_states, gymapi.STATE_POS)

        right_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        right_states['pos'] = right_qpos
        self.gym.set_actor_dof_states(self.env, self.right_handle, right_states, gymapi.STATE_POS)
        
        if self.latency_monitor:
            t_set_dof_end = time.time()
            self.latency_monitor.record_transmit("sim_set_actor_dof_state", t_set_dof_start, t_set_dof_end)
        
        # 物理模拟（计算延迟）
        # 确保时间戳连续
        if self.latency_monitor:
            t_physics_start = t_set_dof_end  # 连续，无间隙
        
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        if self.latency_monitor:
            t_physics_end = time.time()
            self.latency_monitor.record_compute("sim_physics_simulation", t_physics_start, t_physics_end)
        
        # 图形步进和渲染（计算延迟）
        # 确保时间戳连续
        if self.latency_monitor:
            t_render_start = t_physics_end  # 连续，无间隙
        
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        if self.latency_monitor:
            t_render_end = time.time()
            self.latency_monitor.record_compute("sim_viewer_render", t_render_start, t_render_end)
        
        # 相机偏移计算（计算延迟）
        # 确保时间戳连续
        if self.latency_monitor:
            t_cam_offset_start = t_render_end  # 连续，无间隙
        
        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T
        
        if self.latency_monitor:
            t_cam_offset_end = time.time()
            self.latency_monitor.record_compute("sim_camera_offset_computation", t_cam_offset_start, t_cam_offset_end)
        
        # 设置相机位置（传输延迟）
        # 确保时间戳连续
        if self.latency_monitor:
            t_cam_setup_start = t_cam_offset_end  # 连续，无间隙
        
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
            self.latency_monitor.record_transmit("sim_set_camera_location", t_cam_setup_start, t_cam_setup_end)
        
        # 捕获相机图像（传输延迟）
        # 确保时间戳连续
        if self.latency_monitor:
            t_capture_start = t_cam_setup_end  # 连续，无间隙
        
        left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        
        if self.latency_monitor:
            t_capture_end = time.time()
            self.latency_monitor.record_transmit("sim_camera_capture", t_capture_start, t_capture_end)
        
        # 图像处理（计算延迟）
        # 确保时间戳连续
        if self.latency_monitor:
            t_img_process_start = t_capture_end  # 连续，无间隙
        
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]
        
        if self.latency_monitor:
            t_img_process_end = time.time()
            self.latency_monitor.record_compute("sim_image_processing", t_img_process_start, t_img_process_end)

        # Viewer绘制（计算延迟）
        # 确保时间戳连续
        if self.latency_monitor:
            t_draw_start = t_img_process_end  # 连续，无间隙
        
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)
        
        if self.latency_monitor:
            t_draw_end = time.time()
            self.latency_monitor.record_compute("sim_viewer_draw", t_draw_start, t_draw_end)

        return left_image, right_image

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--print-freq', action='store_true', help='Print frame rate')
    parser.add_argument('--latency-log', type=str, default=None, help='Path to latency log file (auto-generated if not specified)')
    parser.add_argument('--network', action='store_true', help='Network mode (for cross-network transmission)')
    args = parser.parse_args()
    
    # 自动生成日志文件名
    if args.latency_log is None:
        args.latency_log = 'latency_log_2d_network.txt' if args.network else 'latency_log_2d_local.txt'
    
    # 创建延迟监控器
    latency_monitor = LatencyMonitor(log_file_path=args.latency_log, network_mode=args.network)
    
    # 获取配置文件路径（相对于脚本所在目录）
    config_path = Path(__file__).parent / 'inspire_hand.yml'
    teleoperator = VuerTeleop(str(config_path), latency_monitor=latency_monitor)
    simulator = Sim(print_freq=args.print_freq, latency_monitor=latency_monitor)

    frame_count = 0
    try:
        while True:
            # 开始新的一帧
            latency_monitor.start_frame()
            loop_start = latency_monitor.current_frame['start_time']  # 记录循环开始时间
            
            # 写入meta共享内存：frame_id(int64)与t0(float64)
            # 注意：hand_move_receive_time和websocket_send_complete_time由TeleVision.py写入
            try:
                mv = memoryview(teleoperator.meta_shm.buf)
                
                # 在写入新的frame_id之前，先读取当前HAND_MOVE接收时间（对应当前帧）
                # 这是处理当前帧时，最近一次HAND_MOVE事件的接收时间
                current_hand_move_time = float(np.frombuffer(mv[16:24], dtype=np.float64, count=1)[0])
                
                # 写入新的frame_id和t0（使用numpy数组方式正确写入共享内存）
                if not hasattr(teleoperator, '_frame_id'):
                    teleoperator._frame_id = 0
                else:
                    teleoperator._frame_id += 1
                frame_id_arr = np.ndarray(1, dtype=np.int64, buffer=mv[:8])
                frame_id_arr[0] = teleoperator._frame_id
                t0_arr = np.ndarray(1, dtype=np.float64, buffer=mv[8:16])
                t0_arr[0] = time.time()
                
                # 如果HAND_MOVE接收时间有效，记录为当前帧的闭环延迟起点
                if current_hand_move_time > 0 and latency_monitor:
                    latency_monitor.current_frame['hand_move_receive_time'] = current_hand_move_time
                    latency_monitor.current_frame['timestamps']['hand_move_receive'] = current_hand_move_time
            except Exception:
                pass
            
            # Teleoperator步骤
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            
            # Simulator步骤
            # 确保时间戳连续：teleoperator.step() 的结束时间应该作为 simulator.step() 的开始时间
            # 但由于是函数调用，我们无法直接获取，所以这里使用当前时间
            # 注意：teleoperator.step() 内部最后记录的是 teleop_hand_retargeting 的结束时间
            # 这个间隙很小（函数调用开销），可以忽略
            left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
            
            # 图像拼接（计算延迟）
            # 确保时间戳连续：simulator.step() 最后记录的是 sim_viewer_draw 的结束时间
            # 这里我们使用当前时间，间隙很小
            if latency_monitor:
                t_stack_start = time.time()
            
            stacked_img = np.hstack((left_img, right_img))
            
            if latency_monitor:
                t_stack_end = time.time()
                latency_monitor.record_compute("image_stack", t_stack_start, t_stack_end)
            
            # 图像传输延迟（写入共享内存）
            # 确保时间戳连续
            if latency_monitor:
                t_img_transmit_start = t_stack_end  # 连续，无间隙
            
            np.copyto(teleoperator.img_array, stacked_img)
            
            if latency_monitor:
                t_img_transmit_end = time.time()
                latency_monitor.record_transmit("image_shared_memory_write", t_img_transmit_start, t_img_transmit_end)
                
                # 记录图像发送完成时间（主机端处理完成）
                latency_monitor.current_frame['image_send_complete_time'] = t_img_transmit_end
                latency_monitor.current_frame['timestamps']['image_send_complete'] = t_img_transmit_end
            
            # 结束当前帧，写入延迟信息
            # end_frame() 会使用当前时间作为结束时间，确保总延迟计算准确
            if latency_monitor:
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
                            current_time = time.time()
                            time_since_hand_move = (current_time - hand_move_time) * 1000  # 毫秒
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
                                
                                # ⚠️ 关键问题：完整的闭环延迟应该包含所有阶段
                                # 完整闭环 = VR设备捕获手部运动 → VR设备显示图像
                                #          = VR→主机传输 + 主机处理 + VR处理 + 主机→VR传输 + VR渲染
                                # 
                                # 当前measured_delay = websocket_complete_time - hand_move_receive_time
                                # 包含：VR→主机传输（在hand_move_receive_time中）+ 主机处理 + VR处理（到WebSocket发送）
                                # 缺少：主机→VR的网络传输 + VR渲染
                                if latency_monitor.network_mode:
                                    # 估算主机->VR的网络传输延迟和VR渲染延迟
                                    estimated_host_to_vr_transmission = 70.0  # 主机→VR网络传输：约70ms
                                    estimated_vr_rendering = 20.0  # VR渲染延迟：约20ms
                                    estimated_host_to_vr_delay = estimated_host_to_vr_transmission + estimated_vr_rendering  # 总计：90ms
                                    
                                    # 分解measured_delay来计算VR->主机的网络传输延迟
                                    # measured_delay = websocket_complete_time - hand_move_receive_time
                                    #                 = VR→主机传输 + 主机处理 + VR处理（到WebSocket发送）
                                    # 
                                    # 主机处理时间 = image_send_complete_time - hand_move_receive_time
                                    # 这准确反映了从主机接收HAND_MOVE事件到图像写入共享内存完成的时间
                                    # VR处理时间 = websocket_complete_time - image_send_complete_time
                                    # 这反映了VR端从共享内存读取到WebSocket发送的时间（不包含网络传输）
                                    host_processing_time = 0.0
                                    vr_processing_time = 0.0
                                    
                                    if 'hand_move_receive_time' in latency_monitor.current_frame and 'image_send_complete_time' in latency_monitor.current_frame:
                                        hand_move_time = latency_monitor.current_frame['hand_move_receive_time']
                                        image_send_time = latency_monitor.current_frame['image_send_complete_time']
                                        
                                        # 主机处理时间：从接收HAND_MOVE到图像写入共享内存完成
                                        if image_send_time >= hand_move_time:
                                            host_processing_time = (image_send_time - hand_move_time) * 1000
                                        else:
                                            # 时间戳异常，使用默认值
                                            host_processing_time = 0.0
                                        
                                        # VR端处理时间（从共享内存读取到WebSocket发送）
                                        # 注意：websocket_complete_time必须大于image_send_time，否则说明时间戳有问题
                                        if websocket_complete_time >= image_send_time:
                                            vr_processing_time = (websocket_complete_time - image_send_time) * 1000
                                        else:
                                            # 时间戳异常（可能是读取了旧的websocket_complete_time），使用默认值
                                            vr_processing_time = 0.0
                                    
                                    # 计算VR->主机的网络传输延迟
                                    # measured_delay = VR→主机传输 + 主机处理 + VR处理
                                    # 所以：VR→主机传输 = measured_delay - 主机处理 - VR处理
                                    calculated_vr_to_host_delay = full_loop_delay - host_processing_time - vr_processing_time
                                    
                                    # 使用计算值和估算值的合理范围（避免负值或不合理的大值）
                                    if calculated_vr_to_host_delay > 0 and calculated_vr_to_host_delay < 500:
                                        estimated_vr_to_host_delay = calculated_vr_to_host_delay
                                    else:
                                        # 如果计算值不合理，假设网络对称
                                        estimated_vr_to_host_delay = estimated_host_to_vr_transmission
                                    
                                    # 检查是否有队列阻塞延迟（WebSocket队列积压导致的额外延迟）
                                    # 队列阻塞延迟应该添加到主机→VR传输延迟中，因为数据需要等待才能发送
                                    # 注意：2D模式可能没有队列阻塞延迟的记录，所以默认为0
                                    queue_blocking_delay = 0.0
                                    if 'queue_blocking_delay_ms' in latency_monitor.current_frame:
                                        queue_blocking_delay = latency_monitor.current_frame['queue_blocking_delay_ms']
                                    elif 'transmit_delays' in latency_monitor.current_frame and 'queue_blocking_delay' in latency_monitor.current_frame['transmit_delays']:
                                        queue_blocking_delay = latency_monitor.current_frame['transmit_delays']['queue_blocking_delay']
                                    
                                    # 实际的主机→VR传输延迟 = 基础传输延迟 + 队列阻塞延迟
                                    actual_host_to_vr_transmission = estimated_host_to_vr_transmission + queue_blocking_delay
                                    
                                    # 完整的闭环延迟 = measured_delay + 主机→VR传输（含队列阻塞）+ VR渲染
                                    complete_full_loop_delay = full_loop_delay + actual_host_to_vr_transmission + estimated_vr_rendering
                                    
                                    latency_monitor.current_frame['measured_delay_ms'] = full_loop_delay  # 重命名：这是不完整的measured延迟
                                    latency_monitor.current_frame['estimated_vr_to_host_transmission_ms'] = estimated_vr_to_host_delay
                                    latency_monitor.current_frame['estimated_host_to_vr_transmission_base_ms'] = estimated_host_to_vr_transmission  # 基础传输延迟
                                    latency_monitor.current_frame['queue_blocking_delay_ms'] = queue_blocking_delay  # 队列阻塞延迟
                                    latency_monitor.current_frame['estimated_host_to_vr_transmission_ms'] = actual_host_to_vr_transmission  # 实际传输延迟（含队列阻塞）
                                    latency_monitor.current_frame['estimated_vr_rendering_ms'] = estimated_vr_rendering
                                    latency_monitor.current_frame['host_processing_time_ms'] = host_processing_time
                                    latency_monitor.current_frame['vr_processing_time_ms'] = vr_processing_time
                                    latency_monitor.current_frame['calculated_vr_to_host_delay_ms'] = calculated_vr_to_host_delay
                                    latency_monitor.current_frame['complete_full_loop_delay_ms'] = complete_full_loop_delay  # 完整的闭环延迟
                                    latency_monitor.current_frame['real_full_loop_delay_ms'] = complete_full_loop_delay  # 保持向后兼容
                                else:
                                    # 本地模式下，同样输出分解：
                                    # measured_delay = VR→主机传输 + 主机处理 + VR处理（到WebSocket发送）
                                    # 主机→VR网络传输与VR渲染近似为0，仅用于分解展示
                                    host_processing_time = latency_monitor.current_frame.get('total_delay_ms', 0)
                                    vr_processing_time = 0.0
                                    if 'image_send_complete_time' in latency_monitor.current_frame:
                                        image_send_time = latency_monitor.current_frame['image_send_complete_time']
                                        if websocket_complete_time >= image_send_time:
                                            vr_processing_time = (websocket_complete_time - image_send_time) * 1000
                                    calculated_vr_to_host_delay = full_loop_delay - host_processing_time - vr_processing_time
                                    if calculated_vr_to_host_delay < 0:
                                        calculated_vr_to_host_delay = 0.0
                                    latency_monitor.current_frame['measured_delay_ms'] = full_loop_delay
                                    latency_monitor.current_frame['estimated_vr_to_host_transmission_ms'] = calculated_vr_to_host_delay
                                    latency_monitor.current_frame['host_processing_time_ms'] = host_processing_time
                                    latency_monitor.current_frame['vr_processing_time_ms'] = vr_processing_time
                                    latency_monitor.current_frame['estimated_host_to_vr_transmission_ms'] = 0.0
                                    latency_monitor.current_frame['estimated_vr_rendering_ms'] = 0.0
                                    latency_monitor.current_frame['complete_full_loop_delay_ms'] = full_loop_delay
                except Exception:
                    pass
                
            latency_monitor.end_frame()
            
            frame_count += 1
    except KeyboardInterrupt:
        # 生成延迟统计摘要
        latency_monitor.generate_summary()
        print(f"\n延迟统计摘要已保存到: {args.latency_log}")
        simulator.end()
        exit(0)
