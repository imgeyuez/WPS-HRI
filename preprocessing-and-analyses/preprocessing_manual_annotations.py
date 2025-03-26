import os
import re
import json
from collections import defaultdict

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
        return {"annotations": [], "text": ""}

    tokens.sort(key=lambda x: x[0])
    full_text_parts = []
    current_pos = 0
    for start, end, token in tokens:
        if current_pos < start:
            gap = " " * (start - current_pos) # adding spaces for gaps in INCEpTION character offsets
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

def collect_all_annotations(root_folder):
    output = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.endswith(".tsv") and "CURATION_USER" in fname:
                fpath = os.path.join(dirpath, fname)
                parsed = parse_annotated_files(fpath)
                relative_path = os.path.relpath(fpath, root_folder)
                base_filename = relative_path.split("/")[0]
                base_filename = base_filename.split(".txt")[0] + ".txt"  # drop everything after .txt

                parsed["filename"] = base_filename
                output.append(parsed)

    return output

# base directory and output file; change these for RoBERTa annotations
base_dir = "../data/curation_backup_2025_03_20/curation"
output_file = "../data/manual_character_annotations.json"

all_annotations = collect_all_annotations(base_dir)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_annotations, f, ensure_ascii=False, indent=2)

print(f"Saved {len(all_annotations)} annotated documents to {output_file}")
