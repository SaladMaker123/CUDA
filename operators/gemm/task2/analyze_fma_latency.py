
from pathlib import Path
import re
import numpy as np

# ================================
# Task 2: FMA Latency Analysis
# ================================
LOG_FILE = (
    Path(__file__).resolve().parent
    / "results"
    / "out.log"
)


def parse_data(text):
    start = "=== Experiment 1: full D x ILP sweep ==="
    end = "=== Experiment 2:"

    if start not in text or end not in text:
        raise RuntimeError("找不到 FMA 参数扫描数据")

    block = text.split(start, 1)[1].split(end, 1)[0]

    rows = []

    for line in block.splitlines():
        match = re.fullmatch(
            r"\s*(\d+)\s+(\d+)\s+([\d.]+)\s*",
            line
        )

        if match:
            rows.append(tuple(map(float, match.groups())))

    if len(rows) != 72:
        raise RuntimeError(
            f"预期 72 组数据，实际解析到 {len(rows)} 组"
        )

    data = np.array(rows)

    return data[:, 0], data[:, 1], data[:, 2]


def linear_fit(y, *features):
    X = np.column_stack([
        np.ones(len(y)), *features
    ])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    predicted = X @ beta

    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1.0 - ss_res / ss_tot

    return beta, r2


def main():
    text = LOG_FILE.read_text(encoding="utf-8")

    D, ILP, C = parse_data(text)

    print("========== FMA 延迟分析 ==========")
    print(f"有效测试数据：{len(C)} 组\n")

    # ================================
    # 1. 加法模型
    # ================================
    beta, r2 = linear_fit(C, D, ILP)

    print("1. 加法模型")
    print("C = b0 + b1*D + b2*ILP")
    print(
        f"b0={beta[0]:.3f}, "
        f"b1={beta[1]:.3f}, "
        f"b2={beta[2]:.3f}, "
        f"R²={r2:.4f}\n"
    )

    # ================================
    # 2. 交互模型
    # ================================
    beta, r2 = linear_fit(C, D, ILP, D * ILP)

    print("2. 交互模型")
    print("C = b0 + b1*D + b2*ILP + b3*D*ILP")
    print(
        f"b0={beta[0]:.3f}, "
        f"b1={beta[1]:.3f}, "
        f"b2={beta[2]:.3f}, "
        f"b3={beta[3]:.3f}, "
        f"R²={r2:.4f}\n"
    )

    # ================================
    # 3. 固定 ILP=1
    # ================================
    mask = ILP == 1

    beta, r2 = linear_fit(
        C[mask], D[mask]
    )

    print("3. 固定 ILP=1")
    print("C = a0 + a1*D")
    print(
        f"a0={beta[0]:.3f}, "
        f"a1={beta[1]:.3f}, "
        f"R²={r2:.4f}"
    )
    print(
        f"D 每增加 2，周期数平均增加 "
        f"{2 * beta[1]:.3f}\n"
    )

    # ================================
    # 4. 固定 D=2
    # ================================
    mask = D == 2

    beta, r2 = linear_fit(
        C[mask], ILP[mask]
    )

    print("4. 固定 D=2")
    print("C = c0 + c1*ILP")
    print(
        f"c0={beta[0]:.3f}, "
        f"c1={beta[1]:.3f}, "
        f"R²={r2:.4f}"
    )
    print(
        f"ILP 每增加 1，周期数平均变化 "
        f"{beta[1]:.3f}"
    )


if __name__ == "__main__":
    main()
