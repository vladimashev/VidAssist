from src.dataset import CrossTaskVideoDataset

dataset = CrossTaskVideoDataset(
    data_root="data/crosstask"
)

print("Total videos:", len(dataset))

sample = dataset[0]

print("\nSample video id:", sample["video_id"])
print("Task id:", sample["task_id"])
print("Task name:", sample["task_name"])
print("Task description:", sample["task_description"])
print("Pred sequence:", sample["pred_sequence"])
print("True sequence:", sample["true_sequence"])