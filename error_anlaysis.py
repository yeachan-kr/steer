import pandas as pd
import jsonlines


# error case 분석
target = "qasc"
is_lora = "lora_True" # default
is_ours = False

df = pd.read_csv(f"./results/llama_confusion_matrix/{target}/{is_ours}/{is_lora}.csv")

our_pred = df["prediction"]
our_pred = our_pred.to_list()
labels = df["labels"]
labels = labels.to_list()

print(our_pred)
print(labels)

err_idx = []
idx = 0
for pred, label in zip(our_pred, labels):
    if float(pred) != float(label):
        err_idx.append(idx)
    idx += 1

result = pd.DataFrame({"idx": err_idx})
result.to_csv(f"./results/llama_confusion_matrix/{target}/error_case/{target}_{is_ours}_error_case.csv", header=True, index=False)


# Llama 1B fine-tuned on target task vs OURS
# df_ours = pd.read_csv(f"./results/llama_confusion_matrix/{target}/error_case/{target}_True_error_case.csv")
# df_base = pd.read_csv(f"./results/llama_confusion_matrix/{target}/error_case/{target}_False_error_case.csv")

# Llama 3B zero-shot on target task vs OURS
df_ours = pd.read_csv(f"./results/llama_confusion_matrix/{target}/error_case/{target}_True_error_case.csv")
df_base = pd.read_csv(f"./results/llama_confusion_matrix/{target}/error_case/{target}_False_error_case_3b.csv")
ours_idx = df_ours["idx"]
ours_idx = ours_idx.to_list()
base_idx = df_base["idx"]
base_idx = base_idx.to_list()

common_list = []
only_base_list = []
only_ours_list = []

for current in base_idx:
    if current in ours_idx:
        common_list.append(int(current))
    else:
        only_base_list.append(int(current))
        

for current in ours_idx:
    if current in common_list:
        continue
    else:
        only_ours_list.append(int(current))

max_length = max(len(common_list), len(only_base_list), len(only_ours_list))
common_list.extend([None] * (max_length - len(common_list)))
only_base_list.extend([None] * (max_length - len(only_base_list)))
only_ours_list.extend([None] * (max_length - len(only_ours_list)))

result = pd.DataFrame({"common": common_list,
                       "only_base_list": only_base_list,
                       "only_ours_list": only_ours_list})
result.to_csv(f"results/llama_confusion_matrix/conq/{target}_analysis_3b.csv", header=True, index=False)
