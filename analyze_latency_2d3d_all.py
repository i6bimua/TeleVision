import os
import re
from statistics import mean


def parse_file(path: str):
    """解析日志，输出：
    - 平均总延迟（主机处理）
    - 平均计算延迟（各计算项求和）
    - 平均传输延迟（各传输项求和）
    - 平均VR→主机传输（若存在“完整闭环延迟分解”段落则解析，否则为0）
    - 样本帧数
    """
    if not os.path.exists(path):
        return None

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_delays = []
    compute_sums = []
    transmit_sums = []
    vr_to_host_vals = []  # 来自“完整闭环延迟分解”中的第1项

    i = 0
    n = len(lines)
    frames = 0

    while i < n:
        if lines[i].startswith('--- 帧 #'):
            frames += 1
            frame_total = None
            frame_compute = 0.0
            frame_transmit = 0.0
            frame_vr_to_host = None

            # 总延迟
            j = i + 1
            while j < n:
                m = re.search(r'^总延迟（主机处理）:\s*([\d.]+)\s*ms', lines[j])
                if m:
                    frame_total = float(m.group(1))
                    break
                if j - i > 30:
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

            # 传输延迟块（本地/网络均可能出现不同标题）
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
                    m = re.search(r'^\s*([A-Za-z0-9_]+):\s*([\d.]+)\s*ms', l)
                    if m:
                        frame_transmit += float(m.group(2))
                    t += 1

            # 完整闭环分解（若存在，解析第1项“VR→主机传输”）
            u = t
            # 回退几行以涵盖分解段落位置
            u_start = max(i, t - 40)
            for p in range(u_start, min(u_start + 120, n)):
                if '完整闭环延迟分解' in lines[p]:
                    # 下一至数行中查找 “1. VR→主机传输: X ms”
                    for q in range(p, min(p + 15, n)):
                        m = re.search(r'VR→主机传输:\s*([\d.]+)\s*ms', lines[q])
                        if m:
                            frame_vr_to_host = float(m.group(1))
                            break
                    break

            if frame_total is not None:
                total_delays.append(frame_total)
                compute_sums.append(frame_compute)
                transmit_sums.append(frame_transmit)
                vr_to_host_vals.append(frame_vr_to_host if frame_vr_to_host is not None else 0.0)

            i = t if t > i else i + 1
        else:
            i += 1

    if not total_delays:
        return None

    avg_total = mean(total_delays)
    avg_compute = mean(compute_sums)
    avg_transmit = mean(transmit_sums)
    avg_vr_to_host = mean(vr_to_host_vals)
    avg_other = avg_total - avg_compute - avg_transmit
    # 为了让百分比合计100%，把“其他延迟”并入主机→VR传输项
    avg_host_to_vr = avg_transmit + (avg_other if avg_other > 0 else 0.0)

    pct = lambda x: (x / avg_total * 100.0) if avg_total > 0 else 0.0

    parts = [
        ('计算延迟', avg_compute),
        ('传输延迟（主机→VR）', avg_host_to_vr),
        ('VR→主机传输', avg_vr_to_host),
    ]
    parts.sort(key=lambda x: x[1], reverse=True)
    dom_name, dom_val = parts[0]

    return {
        'frames': len(total_delays),
        'avg_total': avg_total,
        'avg_compute': avg_compute,
        'avg_vr_to_host': avg_vr_to_host,
        'avg_host_to_vr': avg_host_to_vr,
        'compute_pct': pct(avg_compute),
        'vr_to_host_pct': pct(avg_vr_to_host),
        'host_to_vr_pct': pct(avg_host_to_vr),
        'dominant': f'{dom_name}（{pct(dom_val):.1f}%）',
    }


def print_table(title: str, rows):
    print(f"\n## {title}")
    headers = ['模式', '总延迟', '计算延迟', '传输延迟(VR→主机)', '传输延迟(主机→VR)', '主要影响部分', '样本帧数']
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---'] * len(headers)) + '|')
    for name, r in rows:
        print('| ' + ' | '.join([
            name,
            f"{r['avg_total']:.3f} ms",
            f"{r['avg_compute']:.3f} ms ({r['compute_pct']:.1f}%)",
            f"{r['avg_vr_to_host']:.3f} ms ({r['vr_to_host_pct']:.1f}%)",
            f"{r['avg_host_to_vr']:.3f} ms ({r['host_to_vr_pct']:.1f}%)",
            r['dominant'],
            str(r['frames'])
        ]) + ' |')


def main():
    files = {
        '2D 本地': 'latency_log_2d_local.txt',
        '3D 本地': 'latency_log_panorama_local.txt',
        '2D 网络': 'latency_log_2d_network.txt',
        '3D 网络': 'latency_log_panorama_network.txt',
    }

    local_rows = []
    net_rows = []

    for name, path in files.items():
        r = parse_file(path)
        if r is None:
            continue
        if '本地' in name:
            local_rows.append((name.replace(' 本地', ''), r))
        else:
            net_rows.append((name.replace(' 网络', ''), r))

    if local_rows:
        print_table('本地模式', local_rows)
    if net_rows:
        print_table('网络模式', net_rows)


if __name__ == '__main__':
    main()


