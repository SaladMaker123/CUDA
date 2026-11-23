
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================================
# GPU Device Information
# ================================
@dataclass(frozen=True)
class DeviceInfo:
    name: str
    clock_rate_ghz: float
    multiprocessor_count: int

    smem_per_sm_kb: int
    smem_per_block_kb: int
    registers_per_sm_kb: int
    max_reg_per_thread: int

    warp_size: int
    max_threads_per_block: int
    max_threads_per_sm: int
    max_blocks_per_sm: int

    gmem_bandwidth_tbps: float
    smem_bandwidth_tbps: float
    fma_fp32_tflops: float

    fma_issue_latency_cycles: float
    fma_compute_latency_cycles: float
    mpe_per_mp: int

    inst_cache_size_kb: int
    inst_size_byte: int


# ================================
# RTX 5070 Ti Configuration
# ================================
def RTX5070Ti():
    return DeviceInfo(
        name="NVIDIA RTX 5070 Ti",

        clock_rate_ghz=2.452,
        multiprocessor_count=70,

        smem_per_sm_kb=100,
        smem_per_block_kb=99,
        registers_per_sm_kb=256,
        max_reg_per_thread=255,

        warp_size=32,
        max_threads_per_block=1024,
        max_threads_per_sm=1536,
        max_blocks_per_sm=24,

        gmem_bandwidth_tbps=0.836,
        smem_bandwidth_tbps=6.149,
        fma_fp32_tflops=49.33,

        # Modeling assumptions
        fma_issue_latency_cycles=1.0,

        # TASK2 measured estimate
        fma_compute_latency_cycles=3.47,

        # Scheduler partition model
        mpe_per_mp=4,

        # Instruction cache modeling assumptions
        inst_cache_size_kb=4,
        inst_size_byte=16,
    )


# ================================
# SGEMM Tiling Analysis
# ================================
def analyze(device, BM, BK, TM, dtype=4, extra_reg=20):

    BN = BM
    TN = TM

    outputs_per_thread = TM * TN
    outputs_per_block = BM * BN

    # ================================
    # 1. Thread Constraint
    # ================================
    if outputs_per_block % outputs_per_thread:
        return None, "threads"

    num_threads = outputs_per_block // outputs_per_thread

    if (
        num_threads < device.warp_size
        or num_threads % device.warp_size != 0
        or num_threads > device.max_threads_per_block
    ):
        return None, "threads"

    # ================================
    # 2. Register Constraint
    # ================================
    regs_per_thread = (
        TM * TN + TM + TN + extra_reg
    )

    regs_per_sm = (
        device.registers_per_sm_kb * 1024 // 4
    )

    if (
        regs_per_thread > device.max_reg_per_thread
        or num_threads * regs_per_thread > regs_per_sm
    ):
        return None, "regs"

    # ================================
    # 3. Shared Memory Constraint
    # ================================
    smem_kb = (
        (BM * BK + BN * BK) * dtype / 1024
    )

    if smem_kb > min(
        device.smem_per_sm_kb,
        device.smem_per_block_kb
    ):
        return None, "smem"

    # ================================
    # 4. Instruction Cache Constraint
    # ================================
    budget = (
        device.inst_cache_size_kb * 1024
        // device.inst_size_byte
    )

    if TM * TN * BK > budget:
        return None, "icache"

    # ================================
    # 5. Resident Blocks per SM
    # ================================
    blocks_per_sm = min(
        device.max_threads_per_sm // num_threads,

        regs_per_sm // (
            num_threads * regs_per_thread
        ),

        int(device.smem_per_sm_kb // smem_kb),

        device.max_blocks_per_sm,
    )

    if blocks_per_sm < 1:
        return None, "regs"

    # ================================
    # 6. Pipeline Utilization
    # ================================
    warps_per_sm = (
        blocks_per_sm * num_threads
        // device.warp_size
    )

    latency = max(
        device.fma_issue_latency_cycles,

        device.fma_compute_latency_cycles
        / outputs_per_thread,
    )

    pipeline_util = min(
        1.0,
        warps_per_sm
        / (device.mpe_per_mp * latency)
    )

    # ================================
    # 7. Arithmetic Intensity
    # ================================
    ai_gmem = (
        2.0 * BM * BN
        / ((BM + BN) * dtype)
    )

    ai_smem = (
        2.0 * TM * TN
        / ((TM + TN) * dtype)
    )

    # ================================
    # 8. Roofline Performance
    # ================================
    perf = min(
        pipeline_util * device.fma_fp32_tflops,

        device.gmem_bandwidth_tbps * ai_gmem,

        device.smem_bandwidth_tbps * ai_smem,
    )

    return perf, None


# ================================
# Parameter Sweep
# ================================
def build_heat(device, bmn_list, bk_list, tmn_list):

    heat = np.full(
        (len(bk_list), len(bmn_list)),
        np.nan
    )

    details = [
        [[] for _ in bmn_list]
        for _ in bk_list
    ]

    for bm, bk, tm in product(
        bmn_list, bk_list, tmn_list
    ):
        i = bk_list.index(bk)
        j = bmn_list.index(bm)

        perf, reason = analyze(
            device, bm, bk, tm
        )

        details[i][j].append(
            (tm, perf, reason)
        )

        if perf is not None:
            pct = (
                perf / device.fma_fp32_tflops
                * 100
            )

            if (
                np.isnan(heat[i, j])
                or pct > heat[i, j]
            ):
                heat[i, j] = pct

    return heat, details


# ================================
# Export Results
# ================================
def export_log(
    device, heat, details,
    bmn_list, bk_list, output
):
    lines = [
        f"Device: {device.name}",

        f"FP32 reference: "
        f"{device.fma_fp32_tflops:.2f} TFLOPS",

        f"GMEM: {device.gmem_bandwidth_tbps:.3f} TB/s; "
        f"SMEM: {device.smem_bandwidth_tbps:.3f} TB/s",

        "NOTE: icache budget, issue interval "
        "and scheduler model are assumptions.",

        "",

        "=== Summary: best % of reference FP32 throughput ===",

        f"{'BK':>4} " + "".join(
            f"BM={bm:<7}" for bm in bmn_list
        ),
    ]

    best = []
    valid = 0

    # ================================
    # Summary Table
    # ================================
    for i, bk in enumerate(bk_list):
        cells = []

        for j, bm in enumerate(bmn_list):
            pct = heat[i, j]

            if np.isfinite(pct):
                cells.append(f"{pct:.1f}%")
            else:
                cells.append("invalid")

        lines.append(
            f"{bk:>4} " +
            "".join(f"{cell:<10}" for cell in cells)
        )

    # ================================
    # Detailed Results
    # ================================
    lines.append(
        "\n=== Detail: all BM / BK / TM ==="
    )

    for i, bk in enumerate(bk_list):
        for j, bm in enumerate(bmn_list):

            lines.append(
                f"\nBM=BN={bm}, BK={bk}"
            )

            for tm, perf, reason in details[i][j]:

                if reason is None:
                    valid += 1

                    best.append(
                        (perf, bm, bk, tm)
                    )

                    pct = (
                        perf / device.fma_fp32_tflops
                        * 100
                    )

                    lines.append(
                        f"  TM=TN={tm:<2}  "
                        f"{perf:7.3f} TFLOPS  "
                        f"{pct:5.1f}%"
                    )

                else:
                    lines.append(
                        f"  TM=TN={tm:<2}  "
                        f"invalid ({reason})"
                    )

    # ================================
    # Best Modeled Configurations
    # ================================
    lines.append(
        f"\nValid configurations: {valid}/150"
    )

    if best:
        highest = max(best)[0]

        lines.append(
            f"Best modeled performance: "
            f"{highest:.3f} TFLOPS"
        )

        lines.append(
            "Matching configurations:"
        )

        for perf, bm, bk, tm in sorted(
            best,
            key=lambda x: (x[1], x[2], x[3])
        ):
            if abs(perf - highest) < 1e-8:
                lines.append(
                    f"  BM=BN={bm}, "
                    f"BK={bk}, "
                    f"TM=TN={tm}"
                )

    output.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print(f"Scanned: 150; valid: {valid}")
    print(f"Log saved: {output}")

    if best:
        print(
            f"Best modeled throughput: "
            f"{max(best)[0]:.3f} TFLOPS"
        )


# ================================
# Plot Heatmap
# ================================
def plot_heatmap(
    device, heat,
    bmn_list, bk_list, output
):
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    im = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=100
    )

    ax.set_xticks(
        range(len(bmn_list)),
        labels=bmn_list
    )

    ax.set_yticks(
        range(len(bk_list)),
        labels=bk_list
    )

    ax.set_xlabel("BM = BN")
    ax.set_ylabel("BK")

    ax.set_title(
        "SGEMM Tiling - Estimated Performance / FP32 Reference"
    )

    for i in range(len(bk_list)):
        for j in range(len(bmn_list)):

            pct = heat[i, j]

            label = (
                f"{pct:.0f}%"
                if np.isfinite(pct)
                else "X"
            )

            ax.text(
                j, i, label,
                ha="center",
                va="center",
                fontsize=10
            )

    fig.colorbar(
        im,
        ax=ax,
        label="Percent of reference throughput (%)"
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=160
    )

    plt.close(fig)

    print(f"Plot saved: {output}")


# ================================
# Main
# ================================
def main():

    results = (
        Path(__file__).resolve().parent
        / "results"
    )

    results.mkdir(
        parents=True,
        exist_ok=True
    )

    device = RTX5070Ti()

    bmn_list = [16, 32, 64, 128, 256, 512]
    bk_list = [4, 8, 12, 16, 32]
    tmn_list = [1, 2, 4, 8, 16]

    heat, details = build_heat(
        device,
        bmn_list,
        bk_list,
        tmn_list
    )

    export_log(
        device,
        heat,
        details,
        bmn_list,
        bk_list,
        results / "out.log"
    )

    plot_heatmap(
        device,
        heat,
        bmn_list,
        bk_list,
        results / "heatmap.png"
    )


if __name__ == "__main__":
    main()
