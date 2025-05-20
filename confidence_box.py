# confidence_plot.py
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ────────────────── 1. 전역 글꼴 설정 ──────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 120,
})

# ────────────────── 2. 데이터 정의 ──────────────────
routing_ratios = np.arange(0.1, 1.1, 0.1)          # 0.0 ~ 0.9
# tasks = ["PIQA", "CSQA", "QASC", "STQA"]
tasks = ["CSQA", "QASC", "STQA"]

accuracy_data_llama = {
    # "PIQA": [85.71, 80.00, 62.86, 75.47, 71.55, 72.55, 77.61, 73.28, 83.15, 90.91],
    "CSQA": [80.00, 80.00, 63.16, 59.09, 67.74, 67.75, 74.09, 79.46, 80.52, 91.67],
    "QASC": [0.00, 0.00, 66.67, 50.00, 71.88, 70.03, 74.13, 80.88, 91.14, 94.44],
    "STQA": [50.00, 25.00, 33.33, 58.33, 60.71, 51.39, 61.54, 50.00, 61.54, 85.71],
}
accuracy_data_gpt = {
    # "PIQA": [0.00, 0.00, 0.00, 100.00, 60.47, 81.28, 82.25, 86.60, 76.74, 83.40],
    "CSQA": [0.00, 0.00, 0.00,   0.00, 54.41, 69.59, 74.78, 76.12, 72.00, 79.39],
    "QASC": [0.00, 0.00, 0.00,   0.00, 66.67, 72.83, 78.07, 84.34, 93.75, 86.18],
    "STQA": [0.00, 0.00, 0.00,   0.00,100.00, 59.09, 67.42, 80.77, 61.11, 73.97],
}

# ────────────────── 3. zero-shot 기준선 ──────────────────
# reference_values_llama = {
#     "PIQA": 75.8,  "CSQA": 67.7,  "QASC": 75.9,  "STQA": 60.9,
# }
# reference_values_gpt = {
#     "PIQA": 82.8,  "CSQA": 76.3,  "QASC": 79.0,  "STQA": 68.1,
# }

# reference_values_llama = {
#     "CSQA": 67.7,  "QASC": 75.9,  "STQA": 60.9,
# }
# reference_values_gpt = {
#     "CSQA": 76.3,  "QASC": 79.0,  "STQA": 68.1,
# }


# ────────────────── 4. 그래프 그리기 ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 6))

for idx, task in enumerate(tasks):
    color = f"C{idx}"
    llama_label = [0.44, 0.88, 0.73]
    gpt_label = [0.91, 0.91, 0.77]
    # ── Llama (위쪽) ─────────────────────────
    ax1 = axes[0, idx]
    ax1.bar(
        routing_ratios,
        accuracy_data_llama[task],
        color=color, alpha=0.7, width=0.08,
        label=f"Llama 3.2 (3B) (Corr.: {llama_label[idx]})"                    # ← 추가
    )
    # ax1.axhline(
    #     reference_values_llama[task],
    #     linestyle="--", color="black", linewidth=1.2,
    #     label="Zero-shot (Llama 3.2 (3B))"  # ← 항상 라벨
    # )
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Confidence")
    ax1.set_xticks(routing_ratios)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=9, loc="upper left")                  # ← 경고 사라짐

    # ── ChatGPT (아래쪽) ─────────────────────
    ax2 = axes[1, idx]
    ax2.bar(
        routing_ratios,
        accuracy_data_gpt[task],
        color=color, alpha=0.7, width=0.08,
        label=f"ChatGPT (Corr.: {gpt_label[idx]})" 
    )
    # ax2.axhline(
    #     reference_values_gpt[task],
    #     linestyle="--", color="black", linewidth=1.2,
    #     label="Zero-shot (ChatGPT)"
    # )
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Confidence")
    ax2.set_title(task, y=-0.35, weight="bold")
    ax2.set_xticks(routing_ratios)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9, loc="upper left")

# fig.text(0.005, 0.75, "Llama 3.2 (3B)", va="center", ha="left",
#          fontsize=12, weight="bold")
# fig.text(0.005, 0.28, "ChatGPT",        va="center", ha="left",
#          fontsize=12, weight="bold")

plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
plt.tight_layout()

# ────────────────── 5. 저장 & 표시 ──────────────────
save_path = "./results/figure/confidence_corr.pdf"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()

print(f"✓ 그래프를 '{save_path}' 에 저장했습니다.")
