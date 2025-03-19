import json
import numpy as np
import matplotlib.pyplot as plt
import os

# DIRECTORY
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)

data_path = os.path.join(repo_root, "data", "train_dev_test_data")

train_path = os.path.join(data_path, "train.json")
dev_path = os.path.join(data_path, "dev.json")
test_path = os.path.join(data_path, "test.json")

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


# function to extract actual entities
def extract_entity_counts(json_input):
    '''Extract a dictionary of entity counts from a json file'''
    entity_counts = {}

    for speech in json_input:
        for sentence in speech["sentences"]:
            words = sentence["tokens"]
            labels = sentence["goldlabels"]

            entity = []
            entity_label = None

            for i, label in enumerate(labels):
                if label.startswith("B-"):
                    if entity and entity_label:
                        entity_text = " ".join(entity).lower()
                        entity_counts.setdefault(entity_label, {}).setdefault(entity_text, 0)
                        entity_counts[entity_label][entity_text] += 1

                    entity = [words[i].lower()]
                    entity_label = label[2:]

                elif label.startswith("I-") and entity:
                    entity.append(words[i].lower())

                else:
                    if entity and entity_label:
                        entity_text = " ".join(entity)
                        entity_counts.setdefault(entity_label, {}).setdefault(entity_text, 0)
                        entity_counts[entity_label][entity_text] += 1

                    entity = []
                    entity_label = None

    return entity_counts


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

train_entities = extract_entity_counts(train_data)

for label, entities in train_entities.items():
    print(f"{label}:")
    for entity, count in sorted(entities.items(), key=lambda x: x[1], reverse=True)[:10]:  # Show top 10
        print(f"  {entity}: {count}")


fig, axs = plt.subplots(2, 3, figsize=(18,12))  # 2 rows, 3 columns
axs = axs.flatten()

# Set a maximum of top 10 entities per label
top_n = 10

for i, (label, entities) in enumerate(train_entities.items()):
    if i >= len(axs):
        break

    sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:top_n]

    entities_sorted = [entity for entity, _ in sorted_entities]
    counts_sorted = [count for _, count in sorted_entities]

    axs[i].barh(entities_sorted, counts_sorted, color='lightblue')
    axs[i].set_title(f'Top Entities for {label}')
    axs[i].set_xlabel('Frequency')
    axs[i].set_ylabel('Entities')

plt.tight_layout()
plt.show()


### DO SAME FOR SUBSTRINGS (?)
'''
def extract_token_counts(json_input):
    # List of words to skip (common words like articles, conjunctions, punctuation)
    skip_words = {"the", "a", "and", "of", ",", ".", "(", ")", "to", "in", "on", "for"}

    token_counts = {}

    for speech in json_input:
        for sentence in speech["sentences"]:
            words = sentence["tokens"]
            labels = sentence["goldlabels"]

            for i, label in enumerate(labels):
                token = words[i].lower()  # Get the token and lowercase it

                # Skip unwanted tokens
                if token in skip_words:
                    continue

                if label.startswith("B-"):  # Start of a new entity
                    entity_label = label[2:]  # Extract the entity label (e.g., 'hero', 'victim')

                    # Count the token under the correct entity label
                    token_counts.setdefault(entity_label, {}).setdefault(token, 0)
                    token_counts[entity_label][token] += 1

                elif label.startswith("I-"):  # Continuation of an entity
                    # Count the token under the correct entity label
                    token_counts.setdefault(entity_label, {}).setdefault(token, 0)
                    token_counts[entity_label][token] += 1

                else:  # If it's not part of an entity
                    entity_label = None  # Reset the entity label when no entity is found

    return token_counts


train_tokens  = extract_token_counts(train_data)

for label, entities in train_tokens.items():
    print(f"{label}:")
    for entity, count in sorted(entities.items(), key=lambda x: x[1], reverse=True)[:10]:  # Show top 10
        print(f"  {entity}: {count}")


fig, axs = plt.subplots(2, 3, figsize=(18,12))  # 2 rows, 3 columns
axs = axs.flatten()

# Set a maximum of top 10 entities per label
top_n = 10

for i, (label, entities) in enumerate(train_tokens.items()):
    if i >= len(axs):
        break

    sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:top_n]

    entities_sorted = [entity for entity, _ in sorted_entities]
    counts_sorted = [count for _, count in sorted_entities]

    axs[i].barh(entities_sorted, counts_sorted, color='lightblue')
    axs[i].set_title(f'Top Entities for {label}')
    axs[i].set_xlabel('Frequency')
    axs[i].set_ylabel('Entities')

plt.tight_layout()
plt.show()
'''