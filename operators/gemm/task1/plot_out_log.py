
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ================================
# 文件路径
# ================================
RESULTS = Path(__file__).resolve().parent / "results"

LOG_FILE = RESULTS / "out.log"
OUTPUT_FILE = RESULTS / "out_cache.png"


# ================================
# 读取统一实验日志
# ================================
def read_section(text, start_title, end_title):
    def find_header(title, start=0):
        pattern = r"=+\s*" + re.escape(title) + r"\s*=+"
        return re.search(pattern, text[start:])

    start = find_header(start_title)

    if start is None:
        raise RuntimeError(f"找不到实验数据：{start_title}")

    start_pos = start.end()

    end = find_header(end_title, start_pos)

    if end is None:
        raise RuntimeError(f"找不到实验数据：{end_title}")

    end_pos = start_pos + end.start()

    return text[start_pos:end_pos]


# ================================
# 解析数值数据
# ================================
def read_numeric_rows(text, cols):
    rows = []

    for line in text.splitlines():
        parts = line.split()

        if len(parts) != cols:
            continue

        try:
            row = tuple(float(p) for p in parts)
            rows.append(row)
        except ValueError:
            continue

    return rows


# ================================
# 主程序
# ================================
def main():
    if not LOG_FILE.exists():
        raise FileNotFoundError(
            f"实验日志不存在：{LOG_FILE}"
        )

    log = LOG_FILE.read_text(encoding="utf-8")

    # Cache Latency
    latency_text = read_section(
        log,
        "Cache Latency",
        "L1 Cache Bandwidth"
    )

    # L1 + L2 Bandwidth
    small_text = read_section(
        log,
        "L1 Cache Bandwidth",
        "L2 Cache Bandwidth"
    )

    # L2-only Bandwidth
    large_text = read_section(
        log,
        "L2 Cache Bandwidth",
        "Shared Memory Latency"
    )

    latency_rows = read_numeric_rows(latency_text, 3)
    small_rows = read_numeric_rows(small_text, 3)
    large_rows = read_numeric_rows(large_text, 2)

    if not (latency_rows and small_rows and large_rows):
        raise RuntimeError("实验数据解析失败")

    # ================================
    # 创建图表
    # ================================
    fig, axes = plt.subplots(
        3, 1,
        figsize=(8, 10),
        constrained_layout=True
    )

    # ================================
    # 图 1：Cache Latency
    # ================================
    ax = axes[0]

    ax.plot(
        [r[0] for r in latency_rows],
        [r[1] for r in latency_rows],
        "o-",
        markersize=3
    )

    ax.axvline(
        48 * 1024,
        linestyle="--",
        color="gray",
        label="L2: 48 MiB"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Working Set (KiB)")
    ax.set_ylabel("Latency (ns)")
    ax.set_title("Cache Latency")
    ax.legend()

    # ================================
    # 图 2：L1 + L2 Bandwidth
    # ================================
    ax = axes[1]

    sizes = [r[0] for r in small_rows]

    ax.plot(
        sizes,
        [r[1] / 1000 for r in small_rows],
        "o-",
        label="L1 + L2"
    )

    ax.plot(
        sizes,
        [r[2] / 1000 for r in small_rows],
        "o-",
        label="L2-only"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Working Set (KiB/Block)")
    ax.set_ylabel("Effective Bandwidth (TB/s)")
    ax.set_title("Cache Bandwidth - 70 Blocks")
    ax.legend()

    # ================================
    # 图 3：L2-only Bandwidth
    # ================================
    ax = axes[2]

    ax.plot(
        [r[0] for r in large_rows],
        [r[1] / 1000 for r in large_rows],
        "o-"
    )

    ax.axvline(
        48,
        linestyle="--",
        color="gray",
        label="L2: 48 MiB"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Working Set (MiB)")
    ax.set_ylabel("Effective Bandwidth (TB/s)")
    ax.set_title("L2-only Bandwidth - 140 Blocks")
    ax.legend()

    # ================================
    # 网格与输出
    # ================================
    for ax in axes:
        ax.grid(
            True,
            which="both",
            alpha=0.25
        )

    fig.savefig(OUTPUT_FILE, dpi=160)
    plt.close(fig)

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
