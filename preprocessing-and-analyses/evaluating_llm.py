import os
import json
from collections import defaultdict, Counter

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

    exact_tp = exact_fp = exact_fn = 0
    partial_tp = partial_fp = partial_fn = 0

    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    label_counts = Counter()

    for filename in matching_filenames:
        llm_spans = spans_with_labels(llm_by_filename[filename])
        manual_spans = spans_with_labels(manual_by_filename[filename])

        llm_set = {(s["start"], s["end"], s["label"]) for s in llm_spans}
        manual_set = {(s["start"], s["end"], s["label"]) for s in manual_spans}

        # exact matches
        exact_tp += len(llm_set & manual_set)
        exact_fp += len(llm_set - manual_set)
        exact_fn += len(manual_set - llm_set)

        # partial match
        matched_manual = set()
        matched_llm = set()

        for i, pred in enumerate(llm_spans):
            matched = False
            for j, gold in enumerate(manual_spans):
                if spans_overlap_50(pred, gold):
                    matched_llm.add(i)
                    matched_manual.add(j)
                    label_stats[pred["label"]]["tp"] += 1
                    matched = True
                    break
            if not matched:
                label_stats[pred["label"]]["fp"] += 1

        for j, gold in enumerate(manual_spans):
            if j not in matched_manual:
                label_stats[gold["label"]]["fn"] += 1

        partial_tp += len(matched_llm)
        partial_fp += len(llm_spans) - len(matched_llm)
        partial_fn += len(manual_spans) - len(matched_manual)

        for span in manual_spans:
            label_counts[span["label"]] += 1

    # overall scores
        def calc_scores(tp, fp, fn):
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            return precision, recall, f1

        def compute_label_metrics(label_stats, label_counts):
            label_metrics = {}
            micro_tp = micro_fp = micro_fn = 0
            macro_precision = macro_recall = macro_f1 = 0
            weighted_precision = weighted_recall = weighted_f1 = 0
            total_support = sum(label_counts.values())
            num_labels = len(label_stats)

            for label, stats in label_stats.items():
                tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
                prec, rec, f1 = calc_scores(tp, fp, fn)
                label_metrics[label] = {
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1
                }

                micro_tp += tp
                micro_fp += fp
                micro_fn += fn
                support = label_counts[label]
                weighted_precision += prec * support
                weighted_recall += rec * support
                weighted_f1 += f1 * support
                macro_precision += prec
                macro_recall += rec
                macro_f1 += f1

            macro_avg = {
                "precision": macro_precision / num_labels if num_labels else 0,
                "recall": macro_recall / num_labels if num_labels else 0,
                "f1_score": macro_f1 / num_labels if num_labels else 0,
            }
            weighted_avg = {
                "precision": weighted_precision / total_support if total_support else 0,
                "recall": weighted_recall / total_support if total_support else 0,
                "f1_score": weighted_f1 / total_support if total_support else 0,
            }
            micro_avg = {
                "precision": micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0,
                "recall": micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0,
                "f1_score": (2 * micro_tp) / (2 * micro_tp + micro_fp + micro_fn) if (2 * micro_tp + micro_fp + micro_fn) else 0,
            }

            return label_metrics, micro_avg, macro_avg, weighted_avg

        # exact per-label stats (for consistency, optional but useful)
        exact_label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        exact_label_counts = Counter()

        for filename in matching_filenames:
            llm_spans = spans_with_labels(llm_by_filename[filename])
            manual_spans = spans_with_labels(manual_by_filename[filename])

            llm_set = {(s["start"], s["end"], s["label"]) for s in llm_spans}
            manual_set = {(s["start"], s["end"], s["label"]) for s in manual_spans}

            matched_exact = llm_set & manual_set

            for start, end, label in matched_exact:
                exact_label_stats[label]["tp"] += 1

            for start, end, label in (llm_set - manual_set):
                exact_label_stats[label]["fp"] += 1

            for start, end, label in (manual_set - llm_set):
                exact_label_stats[label]["fn"] += 1

            for span in manual_spans:
                exact_label_counts[span["label"]] += 1

        # compute exact label scores
        exact_precision, exact_recall, exact_f1 = calc_scores(exact_tp, exact_fp, exact_fn)
        exact_label_metrics, exact_micro, exact_macro, exact_weighted = compute_label_metrics(
            exact_label_stats, exact_label_counts
        )

        # compute partial label scores
        partial_precision, partial_recall, partial_f1 = calc_scores(partial_tp, partial_fp, partial_fn)
        partial_label_metrics, partial_micro, partial_macro, partial_weighted = compute_label_metrics(
            label_stats, label_counts
        )

        # store results
        results[fname] = {
            "matching_filenames": len(matching_filenames),
            "exact": {
                "true_positives": exact_tp,
                "false_positives": exact_fp,
                "false_negatives": exact_fn,
                "precision": exact_precision,
                "recall": exact_recall,
                "f1_score": exact_f1,
                "per_label": exact_label_metrics,
                "micro_avg": exact_micro,
                "macro_avg": exact_macro,
                "weighted_avg": exact_weighted
            },
            "partial_overlap": {
                "true_positives": partial_tp,
                "false_positives": partial_fp,
                "false_negatives": partial_fn,
                "precision": partial_precision,
                "recall": partial_recall,
                "f1_score": partial_f1,
                "per_label": partial_label_metrics,
                "micro_avg": partial_micro,
                "macro_avg": partial_macro,
                "weighted_avg": partial_weighted
            }
        }

os.makedirs(output_dir, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
