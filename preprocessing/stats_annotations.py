import json
import numpy as np
import matplotlib.pyplot as plt

# File paths
train_path = r"C:\Users\imgey\Desktop\MASTER_POTSDAM\WiSe2425\PM1_argument_mining\train_dev_test_data\train.json"
dev_path = r"C:\Users\imgey\Desktop\MASTER_POTSDAM\WiSe2425\PM1_argument_mining\train_dev_test_data\dev.json"
test_path = r"C:\Users\imgey\Desktop\MASTER_POTSDAM\WiSe2425\PM1_argument_mining\train_dev_test_data\test.json"

# Load JSON data
def load_json(path):
    with open(path, "r", encoding="UTF8") as f:
        return json.load(f)

train_data = load_json(train_path)
dev_data = load_json(dev_path)
test_data = load_json(test_path)

# Function to count entity frequencies
def count_freqs(json_input):
    freq_dict = {}
    for speech in json_input:
        for sentence in speech["sentences"]:
            for label in sentence["goldlabels"]:
                if label.startswith("B"):
                    l = label[2:]
                    freq_dict[l] = freq_dict.get(l, 0) + 1
    return freq_dict

# Count frequencies in each dataset
train_freq = count_freqs(train_data)
dev_freq = count_freqs(dev_data)
test_freq = count_freqs(test_data)

# Get all unique labels
all_labels = sorted(set(train_freq.keys()) | set(dev_freq.keys()) | set(test_freq.keys()))

# Ensure all labels exist in every dataset (default to 0 if missing)
train_counts = np.array([train_freq.get(label, 0) for label in all_labels])
dev_counts = np.array([dev_freq.get(label, 0) for label in all_labels])
test_counts = np.array([test_freq.get(label, 0) for label in all_labels])

# Calculate total counts and sort labels by total frequency (ascending order)
total_counts = train_counts + dev_counts + test_counts
sorted_indices = np.argsort(total_counts)  # Sort indices by total count

# Apply sorting
all_labels = [all_labels[i] for i in sorted_indices]
train_counts = train_counts[sorted_indices]
dev_counts = dev_counts[sorted_indices]
test_counts = test_counts[sorted_indices]
total_counts = total_counts[sorted_indices]

# Bar positions
y_pos = np.arange(len(all_labels))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(y_pos, train_counts, color='lightblue', label='Train')
ax.barh(y_pos, dev_counts, color='pink', left=train_counts, label='Dev')
ax.barh(y_pos, test_counts, color='salmon', left=train_counts + dev_counts, label='Test')

# Labels and legend
ax.set_yticks(y_pos)
ax.set_yticklabels(all_labels)
ax.set_xlabel('Frequency')
ax.set_title('Label Distribution Across Datasets')
ax.legend()

# Add frequency numbers next to each bar
for i, total in enumerate(total_counts):
    ax.text(total + 5, y_pos[i], str(total), va='center', fontsize=10)

plt.show()
