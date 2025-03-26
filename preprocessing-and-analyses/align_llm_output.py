import os
import json
import difflib
import re

input_dir = "../llm-inference/predictions/formatted-predictions"
output_dir = "../data/llm-annotations"

with open("../data/manual_character_annotations.json", "r") as f:
    manual_data = json.load(f)

manual_texts = [entry["data"]["Text"] for entry in manual_data]

def find_best_match(original_text):
    best_ratio = 0
    best_text = None
    for manual_text in manual_texts:
        ratio = difflib.SequenceMatcher(None, original_text, manual_text).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_text = manual_text
    return best_text if best_ratio > 0.98 else None

def get_offset_map(original, modified):
    diff = list(difflib.ndiff(original, modified))
    offset_map = {}
    orig_idx = 0
    delta = 0

    for d in diff:
        code = d[0]
        if code == ' ':
            offset_map[orig_idx] = delta
            orig_idx += 1
        elif code == '-':
            offset_map[orig_idx] = delta
            orig_idx += 1
            delta -= 1
        elif code == '+':
            delta += 1

    for i in range(orig_idx, len(original)):
        offset_map[i] = delta

    return offset_map

def shift_span(start, end, offset_map):
    return (
        start + offset_map.get(start, 0),
        end + offset_map.get(end - 1, 0)
    )

os.makedirs(output_dir, exist_ok=True)
for filename in os.listdir(input_dir):
    if not filename.endswith(".json"):
        continue

    match = re.match(r"(llama|deepseek)_(zero|few)shot\.json", filename)
    if not match:
        continue

    model, shot_type = match.groups()
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{model}_{shot_type}shot.json")

    with open(input_path, "r", encoding="utf-8") as f:
        llm_data = json.load(f)

    matched_llm_entries = []

    for entry in llm_data:
        original_text = entry["data"]["Text"]
        matched_text = find_best_match(original_text)

        if matched_text:
            offset_map = get_offset_map(original_text, matched_text)

            for ann in entry.get("annotations", []):
                for span in ann["result"]:
                    old_start = span["value"]["start"]
                    old_end = span["value"]["end"]
                    new_start, new_end = shift_span(old_start, old_end, offset_map)

                    span["value"]["start"] = new_start
                    span["value"]["end"] = new_end
                    span["value"]["text"] = matched_text[new_start:new_end]

            entry["data"]["Text"] = matched_text
            matched_llm_entries.append(entry)

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(matched_llm_entries, out_f, indent=2, ensure_ascii=False)