import os
import json

input_dir = "../data/llm-annotations"
output_dir = "../results"
model = ["llama", "deepseek"]
shot_type = ["few", "zero"]
output_path = os.path.join(output_dir, f"llm_results.json")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def spans_with_labels(annotation_list):
    return [
        {"start": r["value"]["start"], "end": r["value"]["end"], "label": r["value"]["labels"][0]}
        for r in annotation_list
    ]

def spans_overlap_50(pred, gold):
    if pred["label"] != gold["label"]:
        return False
    overlap_start = max(pred["start"], gold["start"])
    overlap_end = min(pred["end"], gold["end"])
    overlap_length = max(0, overlap_end - overlap_start)
    gold_length = gold["end"] - gold["start"]
    return gold_length > 0 and (overlap_length / gold_length) > 0.5

files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
results = {}

for fname in files:
    file_path = os.path.join(input_dir, fname)
    data = load_json(file_path)

    llm_by_filename = {
        entry["filename"]: entry["annotations"][0]["result"]
        for entry in data
        if "filename" in entry
    }

    manual_data = load_json("../data/manual_character_annotations.json")
    manual_by_filename = {
        entry["filename"]: entry["annotations"][0]["result"]
        for entry in manual_data
        if "filename" in entry
    }

    matching_filenames = set(llm_by_filename) & set(manual_by_filename)

    # evaluation metrics
    exact_tp = exact_fp = exact_fn = 0
    partial_tp = partial_fp = partial_fn = 0

    for filename in matching_filenames:
        llm_spans = spans_with_labels(llm_by_filename[filename])
        manual_spans = spans_with_labels(manual_by_filename[filename])

        llm_set = {(s["start"], s["end"], s["label"]) for s in llm_spans}
        manual_set = {(s["start"], s["end"], s["label"]) for s in manual_spans}

        # exact matches
        exact_tp += len(llm_set & manual_set)
        exact_fp += len(llm_set - manual_set)
        exact_fn += len(manual_set - llm_set)

        # partial match (has the same label and spans overlap by at least 50%)
        matched_manual = set()
        matched_llm = set()

        for i, pred in enumerate(llm_spans):
            for j, gold in enumerate(manual_spans):
                if spans_overlap_50(pred, gold):
                    matched_llm.add(i)
                    matched_manual.add(j)
                    break

        partial_tp += len(matched_llm)
        partial_fp += len(llm_spans) - len(matched_llm)
        partial_fn += len(manual_spans) - len(matched_manual)

    exact_precision = exact_tp / (exact_tp + exact_fp) if (exact_tp + exact_fp) else 0
    exact_recall = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) else 0
    exact_f1 = 2 * exact_precision * exact_recall / (exact_precision + exact_recall) if (exact_precision + exact_recall) else 0

    partial_precision = partial_tp / (partial_tp + partial_fp) if (partial_tp + partial_fp) else 0
    partial_recall = partial_tp / (partial_tp + partial_fn) if (partial_tp + partial_fn) else 0
    partial_f1 = 2 * partial_precision * partial_recall / (partial_precision + partial_recall) if (partial_precision + partial_recall) else 0

    results[fname] = {
        "matching_filenames": len(matching_filenames),
        "exact": {
            "true_positives": exact_tp,
            "false_positives": exact_fp,
            "false_negatives": exact_fn,
            "precision": exact_precision,
            "recall": exact_recall,
            "f1_score": exact_f1
        },
        "partial_overlap": {
            "true_positives": partial_tp,
            "false_positives": partial_fp,
            "false_negatives": partial_fn,
            "precision": partial_precision,
            "recall": partial_recall,
            "f1_score": partial_f1
        }
    }

os.makedirs(output_dir, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
