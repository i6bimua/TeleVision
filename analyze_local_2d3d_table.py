import re
import sys
from statistics import mean


def parse_local_log(path: str):
    """解析本地模式日志，汇总每帧：总延迟、计算延迟、传输延迟。

    约定：
    - VR→主机传输（本地）未单独记录，近似为0ms。
    - 主机→VR传输包括所有传输延迟项（共享内存写入、相机捕获等）。
    """
    total_delays = []
    compute_sums = []
    transmit_sums = []  # 所有传输延迟的总和
    host_to_vr_all = []  # 所有传输延迟（主机→VR）
    frames = 0

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip('\n')
        if line.startswith('--- 帧 #'):
            frames += 1
            frame_total = None
            frame_compute = 0.0
            frame_transmit = 0.0

            # 总延迟（主机处理）
            j = i + 1
            while j < n:
                l = lines[j].rstrip('\n')
                m = re.search(r'^总延迟（主机处理）:\s*([\d.]+)\s*ms', l)
                if m:
                    frame_total = float(m.group(1))
                    break
                if j - i > 20:
                    break
                j += 1

            # 计算延迟块
            k = j
            while k < n and not lines[k].startswith('计算延迟:'):
                k += 1
            if k < n and lines[k].startswith('计算延迟:'):
                k += 1
                while k < n:
                    l = lines[k].rstrip('\n')
                    if (not l) or l.startswith('传输延迟') or l.startswith('带宽不足导致的延迟:') or l.startswith('--- 帧 #'):
                        break
                    m = re.search(r':\s*([\d.]+)\s*ms', l)
                    if m:
                        frame_compute += float(m.group(1))
                    k += 1

            # 传输延迟块（包括"传输延迟:"和"传输延迟（包括通信延迟）:"）
            t = k
            while t < n and not lines[t].startswith('传输延迟'):
                if lines[t].startswith('带宽不足导致的延迟:') or lines[t].startswith('--- 帧 #'):
                    break
                t += 1
            if t < n and lines[t].startswith('传输延迟'):
                t += 1
                while t < n:
                    l = lines[t].rstrip('\n')
                    if (not l) or l.startswith('带宽不足导致的延迟:') or l.startswith('--- 帧 #'):
                        break
                    # 匹配传输延迟项（可能有缩进）
                    m = re.search(r'^\s*([A-Za-z0-9_]+):\s*([\d.]+)\s*ms', l)
                    if m:
                        val = float(m.group(2))
                        frame_transmit += val
                    t += 1

            if frame_total is not None:
                total_delays.append(frame_total)
                compute_sums.append(frame_compute)
                transmit_sums.append(frame_transmit)
                host_to_vr_all.append(frame_transmit)  # 所有传输延迟都算作主机→VR

            i = t
        else:
            i += 1

    if frames == 0 or len(total_delays) == 0:
        return None

    avg_total = mean(total_delays)
    avg_compute = mean(compute_sums) if compute_sums else 0.0
    avg_transmit = mean(transmit_sums) if transmit_sums else 0.0
    avg_vr_to_host = 0.0  # 本地近似为0
    
    # 其他延迟（未测量时间等）= 总延迟 - 计算延迟 - 传输延迟
    avg_other = avg_total - avg_compute - avg_transmit
    
    # 主机→VR传输延迟 = 所有传输延迟 + 其他延迟（确保百分比加起来是100%）
    avg_host_to_vr = avg_transmit + avg_other

    pct = lambda x: (x / avg_total * 100.0) if avg_total > 0 else 0.0
    
    # 找出主要影响部分（只考虑计算延迟和传输延迟，不包括其他延迟）
    parts = [
        ('计算延迟', avg_compute),
        ('传输延迟（主机→VR）', avg_transmit),
    ]
    parts.sort(key=lambda x: x[1], reverse=True)
    top_name, top_val = parts[0]

    return {
        'frames': frames,
        'avg_total': avg_total,
        'avg_compute': avg_compute,
        'avg_vr_to_host': avg_vr_to_host,
        'avg_host_to_vr': avg_host_to_vr,  # 传输延迟 + 其他延迟，确保百分比加起来是100%
        'avg_other': avg_other,
        'compute_pct': pct(avg_compute),
        'vr_to_host_pct': pct(avg_vr_to_host),
        'host_to_vr_pct': pct(avg_host_to_vr),  # 包含其他延迟，确保是100%
        'other_pct': pct(avg_other),
        'dominant': f'{top_name}（{pct(top_val):.1f}%）',
    }


def format_row(name, r):
    # 总延迟不显示百分比，它是基准值
    # 计算延迟、传输延迟的百分比应该加起来接近100%（加上其他延迟）
    return [
        name,
        f"{r['avg_total']:.3f} ms",  # 总延迟不显示百分比
        f"{r['avg_compute']:.3f} ms ({r['compute_pct']:.1f}%)",
        f"{r['avg_vr_to_host']:.3f} ms ({r['vr_to_host_pct']:.1f}%)",
        f"{r['avg_host_to_vr']:.3f} ms ({r['host_to_vr_pct']:.1f}%)",
        f"{r['dominant']}",
        f"{r['frames']}",
    ]


def main():
    f2d = 'latency_log_2d_local.txt'
    f3d = 'latency_log_panorama_local.txt'
    r2d = parse_local_log(f2d)
    r3d = parse_local_log(f3d)

    if r2d is None:
        print(f"无法解析: {f2d}")
        sys.exit(1)
    if r3d is None:
        print(f"无法解析: {f3d}")
        sys.exit(1)

    headers = ['模式', '总延迟', '计算延迟', '传输延迟(VR→主机)', '传输延迟(主机→VR)', '主要影响部分', '样本帧数']
    rows = [format_row('2D', r2d), format_row('3D', r3d)]

    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main()


