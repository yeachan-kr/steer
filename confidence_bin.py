import matplotlib as mpl
import os
mpl.rcParams.update({
    "font.family": "sans-serif",          # 기본 계열을 sans-serif 로
    "font.sans-serif": ["DejaVu Sans"],   # 1순위 글꼴 지정
    "axes.titlesize": 13,                 # (선택) 제목 크기
    "axes.labelsize": 11,                 # (선택) 축 라벨 크기
})

import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
routing_ratios = [0, 0.1, 0.2, 0.3,0.4, 0.5,0.6, 0.7, 0.8, 0.9, 1.0]
tasks = ["PIQA", "CSQA", "QASC", "STQA"]

# llama
accuracy_data_llama = {
    "PIQA": [85.71,
80.00,
62.86,
75.47,
71.55,
72.55,
77.61,
73.28,
83.15,
90.91],
    "CSQA": [80.00,
80.00,
63.16,
59.09,
67.74,
67.75,
74.09,
79.46,
80.52,
91.67],
    "QASC": [0.00,
0.00,
66.67,
50.00,
71.88,
70.03,
74.13,
80.88,
91.14,
94.44],
    "STQA": [50.00,
25.00,
33.33,
58.33,
60.71,
51.39,
61.54,
50.00,
61.54,
85.71],
}

# gpt
accuracy_data_gpt = {
    "PIQA": [0.00,
0.00,
0.00,
100.00,
60.47,
81.28,
82.25,
86.60,
76.74,
83.40],
    "CSQA": [0.00,
0.00,
0.00,
0.00,
54.41,
69.59,
74.78,
76.12,
72.00,
79.39],
    "QASC": [0.00,
0.00,
0.00,
0.00,
66.67,
72.83,
78.07,
84.34,
93.75,
86.18],
    "STQA": [0.00,
0.00,
0.00,
0.00,
100.00,
59.09,
67.42,
80.77,
61.11,
73.97],
}


# ────────────────── 0. 기준선 값 ──────────────────
reference_values_llama = {
    "PIQA": {"zero-shot": 75.8},
    "CSQA": {"zero-shot": 67.7},
    "QASC": {"zero-shot": 75.9},
    "STQA": {"zero-shot": 60.9},
}
reference_values_gpt = {
    "PIQA": {"zero-shot": 82.8},
    "CSQA": {"zero-shot": 76.3},
    "QASC": {"zero-shot": 79.0},
    "STQA": {"zero-shot": 68.1},
}

# … (전역 설정·데이터 정의는 그대로) …

fig, axes = plt.subplots(2, 4, figsize=(18, 6))

for idx, task in enumerate(tasks):
    color = f"C{idx}"

    # ─ LLAMA (위쪽) ───────────────────────────────
    ax1 = axes[0, idx]
    ax1.bar(routing_ratios, accuracy_data_llama[task],
            color=color, alpha=0.7, width=0.08)

    # ── zero-shot 기준선 ──
    z_llama = reference_values_llama[task]["zero-shot"]
    ax1.axhline(z_llama, linestyle="--", color="black",
                linewidth=1.2,
                label="Zero-shot Baseline" if idx == 0 else None)

    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Uncertainty Threshold")
    ax1.set_title(task, y=-0.35, weight="bold")
    ax1.set_xticks(routing_ratios)
    ax1.set_ylim(0, 105)

    # ─ GPT (아래쪽) ────────────────────────────────
    ax2 = axes[1, idx]
    ax2.bar(routing_ratios, accuracy_data_gpt[task],
            color=color, alpha=0.7, width=0.08)

    z_gpt = reference_values_gpt[task]["zero-shot"]
    ax2.axhline(z_gpt, linestyle="--", color="black",
                linewidth=1.2,
                label="Zero-shot Baseline" if idx == 0 else None)

    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Uncertainty Threshold")
    ax2.set_title(task, y=-0.35, weight="bold")
    ax2.set_xticks(routing_ratios)
    ax2.set_ylim(0, 105)

    # ─ 범례는 각 행의 첫 번째 플롯에만 ─
    if idx == 0:
        ax1.legend(fontsize=9)
        ax2.legend(fontsize=9)

plt.tight_layout()

# ────────────────── 저장 & 출력 ──────────────────
save_path = "./results/figure/confidence.pdf"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()
print(f"✓ 그래프가 '{save_path}' 에 저장되었습니다.")
