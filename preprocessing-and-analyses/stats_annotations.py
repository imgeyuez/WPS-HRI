import json
import math
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
data_path = os.path.join(repo_root, "data", "manual_character_annotations.json")
plot_path = os.path.join(repo_root, "plots", "annotation-stats")
os.makedirs(plot_path, exist_ok=True)

with open(data_path, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

train_data = [item for item in full_data if item['metadata']['split'] == 'train']
dev_data = [item for item in full_data if item['metadata']['split'] == 'dev']
test_data = [item for item in full_data if item['metadata']['split'] == 'test']

# function to count entity frequencies
def count_freqs(json_input):
    freq_dict = {}
    for item in json_input:
        annotations = item["annotations"][0]["result"]
        span_label_map = {}
        for ann in annotations:
            span = (ann["value"]["start"], ann["value"]["end"])
            labels = tuple(sorted(label.upper() for label in ann["value"]["labels"]))
            span_label_map.setdefault(span, set()).update(labels)
        for labels in span_label_map.values():
            label_key = " & ".join(sorted(labels))
            freq_dict[label_key] = freq_dict.get(label_key, 0) + 1
    return freq_dict

# function to extract actual entities
def extract_entity_counts(json_input):
    entity_counts = {}
    for item in json_input:
        annotations = item["annotations"][0]["result"]
        span_label_map = {}
        span_text_map = {}
        for ann in annotations:
            span = (ann["value"]["start"], ann["value"]["end"])
            span_label_map.setdefault(span, set()).update(label.upper() for label in ann["value"]["labels"])
            span_text_map[span] = ann["value"]["text"].lower()
        for span, labels in span_label_map.items():
            label_key = " & ".join(sorted(labels))
            entity = re.sub(r'\s+', ' ', span_text_map[span].strip().lower())
            entity_counts.setdefault(label_key, {})
            entity_counts[label_key][entity] = entity_counts[label_key].get(entity, 0) + 1
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

plt.tight_layout()
plt.savefig(os.path.join(plot_path, "label_distribution.png"))
plt.close()

train_entities = extract_entity_counts(train_data)

for label, entities in train_entities.items():
    print(f"{label}:")
    for entity, count in sorted(entities.items(), key=lambda x: x[1], reverse=True)[:10]:  # Show top 10
        print(f"  {entity}: {count}")


fig, axs = plt.subplots(3, 2, figsize=(14, 16))
axs = axs.flatten()

axs = axs[:5]
fig.delaxes(fig.axes[5])  # delete the 6th unused subplot

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
plt.savefig(os.path.join(plot_path, "top_entities_per_label.png"))
plt.close()


### DO SAME FOR SUBSTRINGS (?)

def extract_token_counts(json_input):
    skip_words = {"the", "a", "and", "of", ",", ".", "(", ")", "to", "in", "on", "for"}
    token_counts = {}

    for speech in json_input:
        text = speech["data"]["Text"].strip()
        annotations = speech["annotations"][0]["result"]

        span_dict = {}
        for ann in annotations:
            span = (ann["value"]["start"], ann["value"]["end"])
            span_dict.setdefault(span, set()).update(label.upper() for label in ann["value"]["labels"])

        for match in re.finditer(r"\b\w[\w'-]*\b", text):
            token = re.sub(r'\s+', ' ', match.group()).strip().lower()
            start, end = match.start(), match.end()

            if token in skip_words:
                continue

            for (span_start, span_end), labels in span_dict.items():
                if start >= span_start and end <= span_end:
                    label_key = " & ".join(sorted(labels))
                    token_counts.setdefault(label_key, {}).setdefault(token, 0)
                    token_counts[label_key][token] += 1
                    break
    return token_counts


train_tokens  = extract_token_counts(train_data)

for label, entities in train_tokens.items():
    print(f"{label}:")
    for entity, count in sorted(entities.items(), key=lambda x: x[1], reverse=True)[:10]:  # Show top 10
        print(f"  {entity}: {count}")


fig, axs = plt.subplots(3, 2, figsize=(14, 16))
axs = axs.flatten()

axs = axs[:5]
fig.delaxes(fig.axes[5])  # delete the 6th unused subplot

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
plt.savefig(os.path.join(plot_path, "top_tokens_per_label.png"))


data_rows = []

for item in full_data:
    text = item["data"]["Text"].strip()
    country = item["metadata"]["country/organization"]
    year = item["metadata"]["year"]
    gender = item["metadata"]["gender"]
    
    # Token count (excluding stopwords)
    tokens = [match.group().strip().lower() for match in re.finditer(r"\b\w[\w'-]*\b", text)]
    tokens = [t for t in tokens]
    num_tokens = len(tokens)

    # Build annotation counts per merged label
    entity_counts = defaultdict(lambda: defaultdict(int))
    annotations = item["annotations"][0]["result"]
    
    span_dict = {}
    for ann in annotations:
        span = (ann["value"]["start"], ann["value"]["end"])
        span_dict.setdefault(span, set()).update(label.upper() for label in ann["value"]["labels"])

    for (start, end), labels in span_dict.items():
        label_key = " & ".join(sorted(labels))
        entity_text = " ".join(text[start:end].lower().split())
        entity_counts[label_key][entity_text] += 1

    data_rows.append({
        "country/organization": country,
        "year": year,
        "gender": gender,
        "only text": text,
        "num_tokens": num_tokens,
        "annotations": dict(entity_counts)
    })

meta_data_filtered = pd.DataFrame(data_rows)

# --- 2. Token count per country for normalization ---

tokens_per_country = meta_data_filtered.groupby("country/organization")["num_tokens"].sum().reset_index()

# --- 3. Extract counts of labeled entities per country ---
rows = []
for _, row in meta_data_filtered.iterrows():
    country = row["country/organization"]
    annotations = row["annotations"]

    for entity_label, entity_dict in annotations.items():
        total_count = sum(entity_dict.values())
        rows.append((country, entity_label, total_count))

entity_df = pd.DataFrame(rows, columns=["country/organization", "entity_label", "count"])
grouped_df = entity_df.groupby(["country/organization", "entity_label"])["count"].sum().reset_index()
grouped_df = grouped_df.merge(tokens_per_country, on="country/organization")

# Normalize per 10 tokens
grouped_df["normalized_count"] = (grouped_df["count"] / grouped_df["num_tokens"]) * 10

# --- 4. Plot ---
pivot_df = grouped_df.pivot(index="country/organization", columns="entity_label", values="normalized_count").fillna(0)
pivot_df.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis", edgecolor="black")

plt.xlabel("Country/Organization")
plt.ylabel("Entity Mentions per 10 Tokens")
plt.title("Normalized Entity Mentions per Country (Stacked by Label)")
plt.xticks(rotation=90)
plt.legend(title="Entity Label", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "normalized_entity_mentions_per_country.png"))
plt.close()


def extract_entities_by_name(df, entity_names, column="annotations"):
    """
    Extracts specific entities from a DataFrame based on whole-word entity names.

    - Uses regex to ensure word-boundary matching (e.g., "men" does not match "women").

    Parameters:
    - df: DataFrame with an 'annotations' column containing nested dictionaries.
    - entity_names: List of entity names (strings) to look for (e.g., ["women", "women's"]).
    - column: Name of the column containing entity annotations (default is 'annotations').

    Returns:
    - DataFrame containing the filtered entities and their details.
    """
    labels = []

    # Compile regex for whole-word matching
    entity_patterns = [rf"\b{re.escape(name)}\b" for name in entity_names]
    entity_regex = re.compile("|".join(entity_patterns), re.IGNORECASE)

    # Iterate through each row and extract relevant entities
    for _, row in df.iterrows():
        annotations = row[column]
        if isinstance(annotations, dict):
            for entity_label, entity_dict in annotations.items():
                for entity, count in entity_dict.items():
                    if entity_regex.search(entity):  # Check for whole-word match
                        labels.append((row["country/organization"], row["year"], row['gender'], entity_label,
                                       entity, count, row['only text'], row["num_tokens"]))

    result_df = pd.DataFrame(labels,
                             columns=["country/organization", "year", "gender", "entity_label", "entity", "count",
                                      'only text', "num_tokens"])

    result_df = result_df.fillna("external")  # Replace NaN values with "external"

    return result_df

women_df = extract_entities_by_name(meta_data_filtered, ["women", "women's"])
men_df = extract_entities_by_name(meta_data_filtered, ["men", "men's"])
un_df = extract_entities_by_name(meta_data_filtered, ["Secretary-General", "resolution", "Council", "Council's", "UN","UN-", "UN's", "Secretary-General's"])

def plot_mentions(df, group_by, normalize_by=None, title=None, colormap="viridis"):
    """
    Plots the mentions of entities in a dataset, with optional normalization.

    Parameters:
    - df: DataFrame containing 'country/organization' or 'year', 'entity_label', 'count', and 'num_tokens'.
    - group_by: Column to group by ('country/organization', 'year', etc).
    - normalize_by: Normalization method ('total' for percentage, 'tokens' for per-token count, None for raw counts).
    - title: Full title for the plot. If None, a default will be generated.
    - colormap: Colormap for the plot.
    """
    aggregated_df = df.groupby([group_by, "entity_label", "entity", "num_tokens"])['count'].sum().reset_index()

    # Normalize
    if normalize_by == "tokens":
        aggregated_df["normalized_count"] = (aggregated_df["count"] / aggregated_df["num_tokens"]) * 10
        ylabel = "Mentions per 10 Tokens"
    elif normalize_by == "total":
        total_counts = aggregated_df.groupby(group_by)['count'].sum().reset_index().rename(
            columns={'count': 'total_count'})
        aggregated_df = aggregated_df.merge(total_counts, on=group_by)
        aggregated_df["normalized_count"] = aggregated_df["count"] / aggregated_df["total_count"] * 100
        ylabel = "Mentions (% of Total)"
    else:
        aggregated_df["normalized_count"] = aggregated_df["count"]
        ylabel = "Raw Mention Counts"

    # Pivot for plotting
    pivot_df = aggregated_df.pivot_table(index=group_by, columns="entity_label", values="normalized_count",
                                         aggfunc="sum", fill_value=0)

    # Plot
    pivot_df.plot(kind="bar", stacked=True, figsize=(12, 6), colormap=colormap, edgecolor="black")
    plt.xlabel(group_by.replace("/organization", "").title())
    plt.ylabel(ylabel)
    plot_title = title if title else f"Mentions by Entity Label across {group_by.title()}"
    plt.title(plot_title)
    plt.xticks(rotation=80)
    plt.legend(title="Entity Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', title.lower()).strip('_')
    filename = f"{safe_title}.png"
    plt.savefig(os.path.join(plot_path, filename))
    plt.close()

#### women mentions
plot_mentions(
    women_df,
    group_by="year",
    normalize_by="tokens",
    title="Mentions of 'Women' and 'Women's' Entities by Year"
)

plot_mentions(
    women_df,
    group_by="year",
    normalize_by="total",
    title="Relative Frequency of 'Women' and 'Women's' Mentions by Year"
)

plot_mentions(
    women_df,
    group_by="country/organization",
    normalize_by="total",
    title="Relative Frequency of 'Women' and 'Women's' Mentions by Country"
)

plot_mentions(
    women_df,
    group_by="country/organization",
    normalize_by="tokens",
    title="Mentions of 'Women' and 'Women's' Entities by Country (per 10 Tokens)"
)

plot_mentions(
    women_df,
    group_by="gender",
    normalize_by="total",
    title="Relative Frequency of 'Women' and 'Women's' Mentions by Speaker Gender"
)

plot_mentions(
    women_df,
    group_by="gender",
    normalize_by="tokens",
    title="Mentions of 'Women' and 'Women's' Entities by Speaker Gender (per 10 Tokens)"
)


### men mentions
plot_mentions(
    men_df,
    group_by="year",
    normalize_by="tokens",
    title="Mentions of 'Men' and 'Men's' Entities by Year"
)

plot_mentions(
    men_df,
    group_by="year",
    normalize_by="total",
    title="Relative Frequency of 'Men' and 'Men's' Mentions by Year"
)

plot_mentions(
    men_df,
    group_by="country/organization",
    normalize_by="total",
    title="Relative Frequency of 'Men' and 'Men's' Mentions by Country"
)

plot_mentions(
    men_df,
    group_by="country/organization",
    normalize_by="tokens",
    title="Mentions of 'Men' and 'Men's' Entities by Country (per 10 Tokens)"
)

plot_mentions(
    men_df,
    group_by="gender",
    normalize_by="total",
    title="Relative Frequency of 'Men' and 'Men's' Mentions by Speaker Gender"
)

plot_mentions(
    men_df,
    group_by="gender",
    normalize_by="tokens",
    title="Mentions of 'Men' and 'Men's' Entities by Speaker Gender (per 10 Tokens)"
)


### UN mentions
plot_mentions(
    un_df,
    group_by="year",
    normalize_by="tokens",
    title="Mentions of UN, Resolution, Council, and Secretary-General by Year"
)

plot_mentions(
    un_df,
    group_by="year",
    normalize_by="total",
    title="Relative Frequency of UN, Resolution, Council, and Secretary-General Mentions by Year"
)

plot_mentions(
    un_df,
    group_by="country/organization",
    normalize_by="total",
    title="Relative Frequency of UN, Resolution, Council, and Secretary-General Mentions by Country"
)

plot_mentions(
    un_df,
    group_by="country/organization",
    normalize_by="tokens",
    title="Mentions of UN, Resolution, Council, and Secretary-General by Country (per 10 Tokens)"
)

plot_mentions(
    un_df,
    group_by="gender",
    normalize_by="total",
    title="Relative Frequency of UN, Resolution, Council, and Secretary-General Mentions by Speaker Gender"
)

plot_mentions(
    un_df,
    group_by="gender",
    normalize_by="tokens",
    title="Mentions of UN, Resolution, Council, and Secretary-General by Speaker Gender (per 10 Tokens)"
)


def plot_mentions_pie(df, entity_strings, title, column="annotations"):
    """
    Pie chart visualization for entity mentions across categories.
    - Supports multiple entity strings (e.g., ["women", "women's"])
    - Uses regex to ensure whole-word matching (e.g., "men" won't match "women").
    - Groups mentions <1.25% into "other".
    """
    category_counts = {}

    # Compile regex pattern for whole-word matching
    entity_patterns = [rf"\b{re.escape(entity)}\b" for entity in entity_strings]
    entity_regex = re.compile("|".join(entity_patterns), re.IGNORECASE)

    for _, row in df.iterrows():
        annotations = row[column]
        if isinstance(annotations, dict):
            for category, entities in annotations.items():  # Iterate categories (HERO, VICTIM, etc.)
                for entity, count in entities.items():  # Iterate entity mentions
                    if entity_regex.search(entity):  # Check for whole-word match
                        if category not in category_counts:
                            category_counts[category] = {}
                        category_counts[category][entity] = category_counts[category].get(entity, 0) + count

    num_categories = len(category_counts)
    rows = math.ceil(num_categories / 2)
    cols = 2 if num_categories > 1 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = axes.flatten() if num_categories > 1 else [axes]

    cmap = plt.get_cmap("viridis")

    for ax, (category, entities) in zip(axes, category_counts.items()):
        total = sum(entities.values())
        threshold = 0.0125 * total  # 1.25% threshold
        filtered_entities = {k: v for k, v in entities.items() if v >= threshold}
        other_total = total - sum(filtered_entities.values())

        if other_total > 0:
            filtered_entities["other"] = other_total

        colors = cmap([i / len(filtered_entities) for i in range(len(filtered_entities))])
        labels = [text if len(text) < 30 else text[:27] + "..." for text in filtered_entities.keys()]
        explode = [0.1 if value / total < 0.1 else 0 for value in filtered_entities.values()]

        ax.pie(
            filtered_entities.values(),
            labels=labels,
            autopct=lambda p: f"{p:.1f}%" if p >= 1 else "",
            startangle=140,
            colors=colors,
            explode=explode,
            wedgeprops={'edgecolor': 'black'},
            textprops={'fontsize': 10},
            labeldistance=1
        )

        for text in ax.texts:
            text.set_rotation(45)  # Adjust the angle as needed

        ax.set_title(category, fontsize=14, fontweight="bold")

    for i in range(num_categories, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle(f"Mentions of Entities Containing {entity_strings} Across Categories")
    safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', title.lower()).strip('_')
    filename = f"{safe_title}.png"
    plt.savefig(os.path.join(plot_path, filename))
    plt.close()

plot_mentions_pie(
    meta_data_filtered,
    ["women", "women's"],
    title="Distribution of 'Women' Mentions Across Entity Categories"
)

plot_mentions_pie(
    meta_data_filtered,
    ["men", "men's"],
    title="Distribution of 'Men' Mentions Across Entity Categories"
)

plot_mentions_pie(
    meta_data_filtered,
    ["Secretary-General", "resolution", "Council's", "UN", "UN-", "UN's", "Secretary-General's", "Council"],
    title="Distribution of UN-Related Mentions Across Entity Categories"
)

