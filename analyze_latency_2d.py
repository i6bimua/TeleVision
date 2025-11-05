#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析2D延迟日志，过滤异常值并进行统计分析
"""

import re
import statistics
from collections import defaultdict
from pathlib import Path

def parse_latency_log(log_file_path):
    """解析延迟日志文件"""
    log_file = Path(log_file_path)
    
    frames = []
    current_frame = None
    in_frame = False
    
    # 异常帧标记
    invalid_frames = set()  # 帧号集合
    extreme_delay_frames = set()  # 极端延迟帧
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    frame_num = 0
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 检测新帧开始
        if line.startswith('--- 帧 #'):
            # 保存上一帧
            if current_frame is not None:
                frames.append(current_frame)
            
            # 开始新帧
            match = re.search(r'帧 #(\d+)', line)
            if match:
                frame_num = int(match.group(1))
                current_frame = {
                    'frame_num': frame_num,
                    'total_delay_ms': None,
                    'fps': None,
                    'compute_delays': {},
                    'transmit_delays': {},
                    'bandwidth_delays': {},
                    'full_loop_delay_ms': None,
                    'full_loop_delay_invalid': False,
                    'has_warning': False,
                    'warning_msg': None
                }
                in_frame = True
        
        # 检测警告
        if '⚠️' in line and '警告' in line:
            if current_frame is not None:
                current_frame['has_warning'] = True
                current_frame['warning_msg'] = line
                invalid_frames.add(frame_num)
        
        # 解析总延迟
        if in_frame and '总延迟（主机处理）:' in line:
            match = re.search(r'总延迟（主机处理）:\s*([\d.]+)\s*ms', line)
            if match:
                current_frame['total_delay_ms'] = float(match.group(1))
        
        # 解析帧率
        if in_frame and '帧率:' in line and 'FPS' in line:
            match = re.search(r'帧率:\s*([\d.]+)\s*FPS', line)
            if match:
                current_frame['fps'] = float(match.group(1))
        
        # 解析计算延迟
        if in_frame and '计算延迟:' in line:
            section = 'compute'
        elif in_frame and '传输延迟:' in line:
            section = 'transmit'
        elif in_frame and '带宽不足导致的延迟:' in line:
            section = 'bandwidth'
        elif in_frame and line and not line.startswith('---') and not line.startswith('总延迟') and not line.startswith('子延迟') and ':' in line:
            # 解析延迟项
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                value_str = parts[1].strip().replace(' ms', '')
                try:
                    value = float(value_str)
                    if section == 'compute':
                        current_frame['compute_delays'][key] = value
                    elif section == 'transmit':
                        current_frame['transmit_delays'][key] = value
                    elif section == 'bandwidth':
                        current_frame['bandwidth_delays'][key] = value
                except ValueError:
                    pass
        
        # 解析闭环延迟（可能在帧内容中，也可能在帧末尾）
        if in_frame and '闭环延迟（端到端）:' in line:
            match = re.search(r'闭环延迟（端到端）:\s*([\d.]+)\s*ms', line)
            if match:
                current_frame['full_loop_delay_ms'] = float(match.group(1))
            # 检查是否标记为无效
            if '无效' in line or 'invalid' in line.lower() or '⚠️' in line:
                current_frame['full_loop_delay_invalid'] = True
                # 注意：这里不添加到invalid_frames，因为这只是闭环延迟无效，不影响总延迟统计
    
    # 保存最后一帧
    if current_frame is not None:
        frames.append(current_frame)
    
    # 识别极端延迟值（使用IQR方法）
    total_delays = [f['total_delay_ms'] for f in frames if f['total_delay_ms'] is not None]
    if total_delays:
        sorted_delays = sorted(total_delays)
        q1_idx = len(sorted_delays) // 4
        q3_idx = 3 * len(sorted_delays) // 4
        q1 = sorted_delays[q1_idx]
        q3 = sorted_delays[q3_idx]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 标记极端值（超过上界或低于下界）
        for f in frames:
            if f['total_delay_ms'] is not None:
                if f['total_delay_ms'] > upper_bound or f['total_delay_ms'] < lower_bound:
                    extreme_delay_frames.add(f['frame_num'])
    
    return frames, invalid_frames, extreme_delay_frames

def filter_normal_frames(frames, invalid_frames, extreme_delay_frames):
    """过滤出正常帧"""
    normal_frames = []
    for f in frames:
        if f['frame_num'] not in invalid_frames and f['frame_num'] not in extreme_delay_frames:
            normal_frames.append(f)
    return normal_frames

def calculate_statistics(frames, normal_frames):
    """计算统计信息"""
    stats = {}
    
    # 总延迟统计
    total_delays = [f['total_delay_ms'] for f in normal_frames if f['total_delay_ms'] is not None]
    if total_delays:
        stats['total_delay'] = {
            'mean': statistics.mean(total_delays),
            'median': statistics.median(total_delays),
            'stdev': statistics.stdev(total_delays) if len(total_delays) > 1 else 0,
            'min': min(total_delays),
            'max': max(total_delays),
            'count': len(total_delays)
        }
    
    # 帧率统计
    fps_list = [f['fps'] for f in normal_frames if f['fps'] is not None]
    if fps_list:
        stats['fps'] = {
            'mean': statistics.mean(fps_list),
            'median': statistics.median(fps_list),
            'stdev': statistics.stdev(fps_list) if len(fps_list) > 1 else 0,
            'min': min(fps_list),
            'max': max(fps_list),
            'count': len(fps_list)
        }
    
    # 闭环延迟统计（只统计有效的）
    valid_loop_delays = [f['full_loop_delay_ms'] for f in normal_frames 
                        if f['full_loop_delay_ms'] is not None and not f['full_loop_delay_invalid']]
    if valid_loop_delays:
        stats['full_loop_delay'] = {
            'mean': statistics.mean(valid_loop_delays),
            'median': statistics.median(valid_loop_delays),
            'stdev': statistics.stdev(valid_loop_delays) if len(valid_loop_delays) > 1 else 0,
            'min': min(valid_loop_delays),
            'max': max(valid_loop_delays),
            'count': len(valid_loop_delays)
        }
    
    # 各模块延迟统计
    compute_stats = defaultdict(list)
    transmit_stats = defaultdict(list)
    bandwidth_stats = defaultdict(list)
    
    for f in normal_frames:
        for key, value in f['compute_delays'].items():
            compute_stats[key].append(value)
        for key, value in f['transmit_delays'].items():
            transmit_stats[key].append(value)
        for key, value in f['bandwidth_delays'].items():
            bandwidth_stats[key].append(value)
    
    stats['compute_modules'] = {}
    for key, values in compute_stats.items():
        if values:
            stats['compute_modules'][key] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
    
    stats['transmit_modules'] = {}
    for key, values in transmit_stats.items():
        if values:
            stats['transmit_modules'][key] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
    
    stats['bandwidth_modules'] = {}
    for key, values in bandwidth_stats.items():
        if values:
            stats['bandwidth_modules'][key] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
    
    return stats

def generate_report(frames, invalid_frames, extreme_delay_frames, normal_frames, stats, output_file):
    """生成分析报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("2D延迟日志统计分析报告（过滤异常值后）\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"总帧数: {len(frames)}\n")
        f.write(f"异常帧数: {len(invalid_frames) + len(extreme_delay_frames)}\n")
        f.write(f"  - 警告帧数（HAND_MOVE时间戳过旧）: {len(invalid_frames)}\n")
        f.write(f"  - 极端延迟帧数（IQR异常值）: {len(extreme_delay_frames)}\n")
        f.write(f"正常帧数: {len(normal_frames)}\n")
        f.write(f"正常帧占比: {len(normal_frames)/len(frames)*100:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("总延迟统计（主机处理）\n")
        f.write("=" * 80 + "\n")
        if 'total_delay' in stats:
            td = stats['total_delay']
            f.write(f"平均值: {td['mean']:.3f} ms\n")
            f.write(f"中位数: {td['median']:.3f} ms\n")
            f.write(f"标准差: {td['stdev']:.3f} ms\n")
            f.write(f"最小值: {td['min']:.3f} ms\n")
            f.write(f"最大值: {td['max']:.3f} ms\n")
            f.write(f"样本数: {td['count']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("帧率统计\n")
        f.write("=" * 80 + "\n")
        if 'fps' in stats:
            fps = stats['fps']
            f.write(f"平均帧率: {fps['mean']:.2f} FPS\n")
            f.write(f"中位帧率: {fps['median']:.2f} FPS\n")
            f.write(f"标准差: {fps['stdev']:.2f} FPS\n")
            f.write(f"最低帧率: {fps['min']:.2f} FPS\n")
            f.write(f"最高帧率: {fps['max']:.2f} FPS\n")
            f.write(f"样本数: {fps['count']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("闭环延迟统计（端到端，仅有效帧）\n")
        f.write("=" * 80 + "\n")
        if 'full_loop_delay' in stats:
            fld = stats['full_loop_delay']
            f.write(f"平均值: {fld['mean']:.3f} ms\n")
            f.write(f"中位数: {fld['median']:.3f} ms\n")
            f.write(f"标准差: {fld['stdev']:.3f} ms\n")
            f.write(f"最小值: {fld['min']:.3f} ms\n")
            f.write(f"最大值: {fld['max']:.3f} ms\n")
            f.write(f"样本数: {fld['count']}\n")
            f.write("注意: 真正的显示延迟还包括VR端渲染时间（通常+10-20ms）\n\n")
        else:
            f.write("无有效闭环延迟数据\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("计算延迟模块统计\n")
        f.write("=" * 80 + "\n")
        if 'compute_modules' in stats:
            for key, module_stats in sorted(stats['compute_modules'].items(), 
                                           key=lambda x: x[1]['mean'], reverse=True):
                f.write(f"\n{key}:\n")
                f.write(f"  平均值: {module_stats['mean']:.3f} ms\n")
                f.write(f"  中位数: {module_stats['median']:.3f} ms\n")
                f.write(f"  标准差: {module_stats['stdev']:.3f} ms\n")
                f.write(f"  最小值: {module_stats['min']:.3f} ms\n")
                f.write(f"  最大值: {module_stats['max']:.3f} ms\n")
                f.write(f"  累计: {module_stats['mean'] * module_stats['count']:.3f} ms\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("传输延迟模块统计\n")
        f.write("=" * 80 + "\n")
        if 'transmit_modules' in stats:
            for key, module_stats in sorted(stats['transmit_modules'].items(), 
                                           key=lambda x: x[1]['mean'], reverse=True):
                f.write(f"\n{key}:\n")
                f.write(f"  平均值: {module_stats['mean']:.3f} ms\n")
                f.write(f"  中位数: {module_stats['median']:.3f} ms\n")
                f.write(f"  标准差: {module_stats['stdev']:.3f} ms\n")
                f.write(f"  最小值: {module_stats['min']:.3f} ms\n")
                f.write(f"  最大值: {module_stats['max']:.3f} ms\n")
                f.write(f"  累计: {module_stats['mean'] * module_stats['count']:.3f} ms\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("带宽延迟模块统计\n")
        f.write("=" * 80 + "\n")
        if 'bandwidth_modules' in stats and stats['bandwidth_modules']:
            for key, module_stats in sorted(stats['bandwidth_modules'].items(), 
                                           key=lambda x: x[1]['mean'], reverse=True):
                f.write(f"\n{key}:\n")
                f.write(f"  平均值: {module_stats['mean']:.3f} ms\n")
                f.write(f"  中位数: {module_stats['median']:.3f} ms\n")
                f.write(f"  标准差: {module_stats['stdev']:.3f} ms\n")
                f.write(f"  最小值: {module_stats['min']:.3f} ms\n")
                f.write(f"  最大值: {module_stats['max']:.3f} ms\n")
                f.write(f"  累计: {module_stats['mean'] * module_stats['count']:.3f} ms\n")
        else:
            f.write("无带宽延迟数据\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("延迟占比分析\n")
        f.write("=" * 80 + "\n")
        
        # 计算总时间和占比（注意：这里只统计主机处理延迟，不包括闭环延迟）
        total_compute_time = sum(sum(f['compute_delays'].values()) for f in normal_frames)
        total_transmit_time = sum(sum(f['transmit_delays'].values()) for f in normal_frames)
        total_bandwidth_time = sum(sum(f['bandwidth_delays'].values()) for f in normal_frames)
        total_frame_time = sum(f['total_delay_ms'] for f in normal_frames if f['total_delay_ms'] is not None)
        
        if total_frame_time > 0:
            f.write(f"总计算时间: {total_compute_time:.3f} ms ({total_compute_time/total_frame_time*100:.2f}%)\n")
            f.write(f"总传输时间: {total_transmit_time:.3f} ms ({total_transmit_time/total_frame_time*100:.2f}%)\n")
            f.write(f"总带宽延迟: {total_bandwidth_time:.3f} ms ({total_bandwidth_time/total_frame_time*100:.2f}%)\n")
            f.write(f"总帧时间（主机处理）: {total_frame_time:.3f} ms\n")
            f.write("\n注意: 上述占比基于主机处理延迟。闭环延迟（端到端）已在上面单独统计。\n\n")
            
            f.write("各计算模块占比:\n")
            compute_module_totals = defaultdict(float)
            for frame in normal_frames:
                for key, value in frame['compute_delays'].items():
                    compute_module_totals[key] += value
            for key, total in sorted(compute_module_totals.items(), key=lambda x: x[1], reverse=True):
                percentage = total / total_frame_time * 100
                f.write(f"  {key}: {percentage:.2f}%\n")
            
            f.write("\n各传输模块占比:\n")
            transmit_module_totals = defaultdict(float)
            for frame in normal_frames:
                for key, value in frame['transmit_delays'].items():
                    transmit_module_totals[key] += value
            for key, total in sorted(transmit_module_totals.items(), key=lambda x: x[1], reverse=True):
                percentage = total / total_frame_time * 100
                f.write(f"  {key}: {percentage:.2f}%\n")

def main():
    log_file = Path('latency_log_2d_local.txt')
    output_file = Path('latency_log_2d_local_analysis.txt')
    
    print("正在解析延迟日志...")
    frames, invalid_frames, extreme_delay_frames = parse_latency_log(log_file)
    
    print(f"总帧数: {len(frames)}")
    print(f"异常帧数: {len(invalid_frames) + len(extreme_delay_frames)}")
    print(f"  - 警告帧数: {len(invalid_frames)}")
    print(f"  - 极端延迟帧数: {len(extreme_delay_frames)}")
    
    print("\n正在过滤正常帧...")
    normal_frames = filter_normal_frames(frames, invalid_frames, extreme_delay_frames)
    print(f"正常帧数: {len(normal_frames)}")
    
    print("\n正在计算统计信息...")
    stats = calculate_statistics(frames, normal_frames)
    
    print("\n正在生成报告...")
    generate_report(frames, invalid_frames, extreme_delay_frames, normal_frames, stats, output_file)
    
    print(f"\n分析完成！报告已保存到: {output_file}")

if __name__ == '__main__':
    main()
