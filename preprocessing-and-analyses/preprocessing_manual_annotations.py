import os
import re
import json
import pandas as pd
from collections import defaultdict

repo_root = ".."
train_path = os.path.join(repo_root, "data", "train_dev_test_split", "train.json")
dev_path = os.path.join(repo_root, "data", "train_dev_test_split", "dev.json")
test_path = os.path.join(repo_root, "data", "train_dev_test_split", "test.json")

# Load train/dev/test to map filenames to splits
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(dev_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)
with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

filename_to_split = {}
for d in train_data:
    filename_to_split[d["filename"]] = "train"
for d in dev_data:
    filename_to_split[d["filename"]] = "dev"
for d in test_data:
    filename_to_split[d["filename"]] = "test"

def parse_annotated_files(filepath):
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    tokens = []
    current_annotations = defaultdict(list)

    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split("\t")
            if len(parts) < 4:
                continue

            token_id, offset_range, token, annotation_field = parts
            start, end = map(int, offset_range.split("-"))
            tokens.append((start, end, token))

            ann_list = annotation_field.split("|") if annotation_field != "_" else []
            for ann in ann_list:
                match_full = re.match(r"([A-Z]+)\[(\d+)\]", ann)
                if match_full:
                    label, group_id = match_full.groups()
                    group_id = int(group_id)
                else:
                    match_simple = re.match(r"([A-Z]+)", ann)
                    if match_simple:
                        label = match_simple.group(1)
                        group_id = f"{label}_{start}"

                if match_full or match_simple:
                    current_annotations[(label, group_id)].append((start, end, token))

    # reconstructing full text from tokens; keeping INCEpTION's character offsets
    if not tokens:
        return {"annotations": [], "data": {"Text": ""}}

    tokens.sort(key=lambda x: x[0])
    full_text_parts = []
    current_pos = 0
    for start, end, token in tokens:
        if current_pos < start:
            gap = " " * (start - current_pos)
            full_text_parts.append(gap)
        full_text_parts.append(token)
        current_pos = end

    full_text = "".join(full_text_parts)

    result_list = []
    for (label, group_id), group_tokens in current_annotations.items():
        group_tokens.sort(key=lambda x: x[0])
        span_start = group_tokens[0][0]
        span_end = group_tokens[-1][1]
        span_text = full_text[span_start:span_end]
        result_list.append({
            "value": {
                "start": span_start,
                "end": span_end,
                "text": span_text,
                "labels": [label.capitalize()]
            }
        })

    return {
        "data": {"Text": full_text},
        "annotations": [{
            "result": result_list,
            "was_cancelled": False
        }]
    }

def collect_all_annotations(root_folder, metadata_df):
    output = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.endswith(".tsv") and "CURATION_USER" in fname:
                fpath = os.path.join(dirpath, fname)
                parsed = parse_annotated_files(fpath)

                relative_path = os.path.relpath(fpath, root_folder)
                base_filename = relative_path.split("/")[0]
                base_filename = base_filename.split(".txt")[0] + ".txt"
                parsed["filename"] = base_filename

                meta_row = metadata_df.loc[metadata_df["filename"] == base_filename]
                metadata = {}
                if not meta_row.empty:
                    row = meta_row.iloc[0]
                    metadata = {
                        "year": str(row.get("year")) if not pd.isna(row.get("year")) else "",
                        "speaker": str(row.get("speaker")) if not pd.isna(row.get("speaker")) else "",
                        "country/organization": str(row.get("country/organization")) if not pd.isna(row.get("country/organization")) else "",
                        "language": str(row.get("language")) if not pd.isna(row.get("language")) else "",
                        "gender": str(row.get("gender")) if not pd.isna(row.get("gender")) else ""
                    }

                metadata["split"] = filename_to_split.get(base_filename, "unknown")

                parsed["metadata"] = metadata
                output.append(parsed)

    return output

base_dir = "../data/curation_backup_2025_03_20/curation"
output_file = "../data/manual_character_annotations.json"
csv_metadata_path = "../data/wps_speeches.csv"

metadata_df = pd.read_csv(csv_metadata_path)
all_annotations = collect_all_annotations(base_dir, metadata_df)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_annotations, f, ensure_ascii=False, indent=2)

print(f"Saved {len(all_annotations)} annotated speeches to {output_file}")