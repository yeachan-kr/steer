import pandas as pd
import matplotlib.pyplot as plt

# 데이터 입력
data = {
    "knowledge_ratio": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
    "PIQA": [82.10, 83.13, 82.64, 82.75, 82.64, 83.20, 82.64],
    "OBQA": [77.60, 77.60, 77.40, 78.80, 79.60, 80.80, 76.60],
}

df = pd.DataFrame(data)

# Zero-shot 성능
piqa_zero_shot = 75.8
obqa_zero_shot = 72.2
piqa_fine = 87.7
obqa_fine = 82.2

# 그래프 그리기
plt.figure(figsize=(8, 6))

# 실선 그래프
plt.plot(df["knowledge_ratio"], df["PIQA"], label="PIQA", marker="o", color="blue")
plt.plot(df["knowledge_ratio"], df["OBQA"], label="OBQA", marker="s", color="orange")

# Zero-shot 수평 파선 추가
plt.axhline(y=piqa_zero_shot, color='blue', linestyle='--', label="PIQA Zero-shot")
plt.axhline(y=obqa_zero_shot, color='orange', linestyle='--', label="OBQA Zero-shot")
plt.axhline(y=piqa_fine, color='skyblue', linestyle='--', label="PIQA Zero-shot")
plt.axhline(y=obqa_fine, color='red', linestyle='--', label="OBQA Zero-shot")

# 축 제목, 그래프 제목
plt.xlabel("Confidence Threshold")
plt.ylabel("Accuracy (%)")

# 범례 추가
plt.legend()

# 격자 추가
plt.grid(True)

# 레이아웃 정리
plt.tight_layout()

# 그래프 출력
plt.show()
plt.savefig("./results/figure/ablation.png")