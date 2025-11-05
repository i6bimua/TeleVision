"""
对比分析两个延迟日志文件：
- latencylogmove: 机器人手臂剧烈运动
- latency_log.txt: 正常运动

分析剧烈运动是否引入额外延迟和帧率降低
"""

import re
import numpy as np

def parse_latency_file(filename):
    """解析延迟日志文件，提取统计数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取统计摘要部分
    stats = {}
    
    # 提取总帧数
    frame_match = re.search(r'总帧数:\s*(\d+)', content)
    if frame_match:
        stats['total_frames'] = int(frame_match.group(1))
    
    # 提取各个延迟项的统计信息
    compute_section = re.search(r'=== 计算延迟统计 ===(.*?)=== 传输延迟统计 ===', content, re.DOTALL)
    if compute_section:
        # 解析每个计算延迟项
        pattern = r'([\w_]+):\s*平均值:\s*([\d.]+)\s*ms.*?标准差:\s*([\d.]+)\s*ms.*?最小值:\s*([\d.]+)\s*ms.*?最大值:\s*([\d.]+)\s*ms'
        matches = re.finditer(pattern, compute_section.group(1), re.DOTALL)
        for match in matches:
            name = match.group(1)
            stats[name] = {
                'mean': float(match.group(2)),
                'std': float(match.group(3)),
                'min': float(match.group(4)),
                'max': float(match.group(5))
            }
    
    # 提取传输延迟统计
    transmit_section = re.search(r'=== 传输延迟统计 ===(.*?)(?:=== 总体延迟统计 ===|$)', content, re.DOTALL)
    if transmit_section:
        pattern = r'([\w_]+):\s*平均值:\s*([\d.]+)\s*ms.*?标准差:\s*([\d.]+)\s*ms.*?最小值:\s*([\d.]+)\s*ms.*?最大值:\s*([\d.]+)\s*ms'
        matches = re.finditer(pattern, transmit_section.group(1), re.DOTALL)
        for match in matches:
            name = match.group(1)
            stats[name] = {
                'mean': float(match.group(2)),
                'std': float(match.group(3)),
                'min': float(match.group(4)),
                'max': float(match.group(5))
            }
    
    # 提取总体延迟统计
    total_section = re.search(r'=== 总体延迟统计 ===(.*?)(?:=== 帧率统计 ===|$)', content, re.DOTALL)
    if total_section:
        total_delay_match = re.search(r'总延迟平均值:\s*([\d.]+)\s*ms', total_section.group(1))
        if total_delay_match:
            stats['total_delay_mean'] = float(total_delay_match.group(1))
        
        total_delay_std_match = re.search(r'总延迟标准差:\s*([\d.]+)\s*ms', total_section.group(1))
        if total_delay_std_match:
            stats['total_delay_std'] = float(total_delay_std_match.group(1))
        
        total_delay_min_match = re.search(r'总延迟最小值:\s*([\d.]+)\s*ms', total_section.group(1))
        if total_delay_min_match:
            stats['total_delay_min'] = float(total_delay_min_match.group(1))
        
        total_delay_max_match = re.search(r'总延迟最大值:\s*([\d.]+)\s*ms', total_section.group(1))
        if total_delay_max_match:
            stats['total_delay_max'] = float(total_delay_max_match.group(1))
    
    # 提取帧率统计
    fps_section = re.search(r'=== 帧率统计 ===(.*?)(?:=== 延迟占比分析 ===|$)', content, re.DOTALL)
    if fps_section:
        fps_mean_match = re.search(r'平均帧率:\s*([\d.]+)\s*FPS', fps_section.group(1))
        if fps_mean_match:
            stats['fps_mean'] = float(fps_mean_match.group(1))
        
        fps_std_match = re.search(r'帧率标准差:\s*([\d.]+)\s*FPS', fps_section.group(1))
        if fps_std_match:
            stats['fps_std'] = float(fps_std_match.group(1))
        
        fps_min_match = re.search(r'最低帧率:\s*([\d.]+)\s*FPS', fps_section.group(1))
        if fps_min_match:
            stats['fps_min'] = float(fps_min_match.group(1))
        
        fps_max_match = re.search(r'最高帧率:\s*([\d.]+)\s*FPS', fps_section.group(1))
        if fps_max_match:
            stats['fps_max'] = float(fps_max_match.group(1))
    
    return stats

def compare_stats(stats_normal, stats_intense):
    """对比两个统计数据"""
    print("=" * 80)
    print("延迟对比分析：正常运动 vs 剧烈运动")
    print("=" * 80)
    
    # 总体对比
    print("\n【总体性能对比】")
    print("-" * 80)
    print(f"正常运动:")
    print(f"  总帧数: {stats_normal.get('total_frames', 'N/A')}")
    print(f"  平均总延迟: {stats_normal.get('total_delay_mean', 'N/A'):.2f} ms")
    print(f"  平均帧率: {stats_normal.get('fps_mean', 'N/A'):.2f} FPS")
    print(f"  最低帧率: {stats_normal.get('fps_min', 'N/A'):.2f} FPS")
    
    print(f"\n剧烈运动:")
    print(f"  总帧数: {stats_intense.get('total_frames', 'N/A')}")
    print(f"  平均总延迟: {stats_intense.get('total_delay_mean', 'N/A'):.2f} ms")
    print(f"  平均帧率: {stats_intense.get('fps_mean', 'N/A'):.2f} FPS")
    print(f"  最低帧率: {stats_intense.get('fps_min', 'N/A'):.2f} FPS")
    
    # 计算差异
    if 'total_delay_mean' in stats_normal and 'total_delay_mean' in stats_intense:
        delay_diff = stats_intense['total_delay_mean'] - stats_normal['total_delay_mean']
        delay_percent = (delay_diff / stats_normal['total_delay_mean']) * 100
        print(f"\n差异:")
        print(f"  总延迟差异: {delay_diff:+.2f} ms ({delay_percent:+.1f}%)")
    
    if 'fps_mean' in stats_normal and 'fps_mean' in stats_intense:
        fps_diff = stats_intense['fps_mean'] - stats_normal['fps_mean']
        fps_percent = (fps_diff / stats_normal['fps_mean']) * 100
        print(f"  帧率差异: {fps_diff:+.2f} FPS ({fps_percent:+.1f}%)")
    
    # 详细模块对比
    print("\n【各模块延迟对比】")
    print("-" * 80)
    
    # 关键模块列表
    key_modules = [
        'teleop_preprocessor',
        'teleop_build_left_pose',
        'teleop_build_right_pose',
        'teleop_hand_retargeting',
        'sim_physics_simulation',
        'panorama_camera_capture',
        'panorama_horizon_conversion',
        'panorama_equirectangular_conversion',
        'sim_viewer_render',
        'image_shared_memory_write'
    ]
    
    print(f"{'模块':<40} {'正常运动(ms)':<15} {'剧烈运动(ms)':<15} {'差异(ms)':<12} {'差异(%)':<10}")
    print("-" * 92)
    
    for module in key_modules:
        normal_val = stats_normal.get(module, {}).get('mean', None)
        intense_val = stats_intense.get(module, {}).get('mean', None)
        
        if normal_val is not None and intense_val is not None:
            diff = intense_val - normal_val
            diff_percent = (diff / normal_val) * 100 if normal_val > 0 else 0
            print(f"{module:<40} {normal_val:<15.3f} {intense_val:<15.3f} {diff:>+11.3f} {diff_percent:>+9.1f}%")
    
    # 分析剧烈运动的影响
    print("\n【剧烈运动影响分析】")
    print("-" * 80)
    
    # 检查哪些模块受剧烈运动影响最大
    affected_modules = []
    for module in key_modules:
        normal_val = stats_normal.get(module, {}).get('mean', None)
        intense_val = stats_intense.get(module, {}).get('mean', None)
        
        if normal_val is not None and intense_val is not None and normal_val > 0:
            diff_percent = ((intense_val - normal_val) / normal_val) * 100
            if abs(diff_percent) > 5:  # 超过5%的变化
                affected_modules.append((module, diff_percent, intense_val - normal_val))
    
    if affected_modules:
        affected_modules.sort(key=lambda x: abs(x[1]), reverse=True)
        print("受剧烈运动影响最大的模块（按变化百分比排序）:")
        for module, diff_percent, diff_ms in affected_modules[:10]:
            print(f"  {module:<40} {diff_percent:>+7.1f}% ({diff_ms:>+7.3f} ms)")
    else:
        print("未发现明显受影响的模块（变化<5%）")
    
    # 稳定性分析（标准差对比）
    print("\n【稳定性分析（标准差对比）】")
    print("-" * 80)
    print(f"{'模块':<40} {'正常运动标准差(ms)':<20} {'剧烈运动标准差(ms)':<22} {'差异':<10}")
    print("-" * 92)
    
    for module in key_modules:
        normal_std = stats_normal.get(module, {}).get('std', None)
        intense_std = stats_intense.get(module, {}).get('std', None)
        
        if normal_std is not None and intense_std is not None:
            diff_std = intense_std - normal_std
            print(f"{module:<40} {normal_std:<20.3f} {intense_std:<22.3f} {diff_std:>+9.3f}")
    
    # 结论
    print("\n【结论】")
    print("-" * 80)
    
    if 'fps_mean' in stats_normal and 'fps_mean' in stats_intense:
        fps_drop = stats_normal['fps_mean'] - stats_intense['fps_mean']
        if fps_drop > 1:
            print(f"⚠️  剧烈运动导致帧率下降 {fps_drop:.2f} FPS")
        else:
            print(f"✓  剧烈运动对帧率影响较小（差异 < 1 FPS）")
    
    if 'total_delay_mean' in stats_normal and 'total_delay_mean' in stats_intense:
        delay_increase = stats_intense['total_delay_mean'] - stats_normal['total_delay_mean']
        if delay_increase > 5:
            print(f"⚠️  剧烈运动导致总延迟增加 {delay_increase:.2f} ms")
        else:
            print(f"✓  剧烈运动对总延迟影响较小（差异 < 5 ms）")
    
    # 找出瓶颈模块
    print("\n【主要瓶颈模块】")
    print("-" * 80)
    print("剧烈运动场景下的主要延迟来源:")
    intense_modules = [(k, v['mean']) for k, v in stats_intense.items() 
                      if isinstance(v, dict) and 'mean' in v and k not in ['total_delay_mean', 'fps_mean']]
    intense_modules.sort(key=lambda x: x[1], reverse=True)
    for module, avg_delay in intense_modules[:5]:
        print(f"  {module:<40} {avg_delay:>7.2f} ms")

if __name__ == '__main__':
    print("解析延迟日志文件...")
    stats_normal = parse_latency_file('latency_log.txt')
    stats_intense = parse_latency_file('latencylogmove')
    
    print(f"\n正常运动场景解析完成，总帧数: {stats_normal.get('total_frames', 'N/A')}")
    print(f"剧烈运动场景解析完成，总帧数: {stats_intense.get('total_frames', 'N/A')}")
    
    compare_stats(stats_normal, stats_intense)
    
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

