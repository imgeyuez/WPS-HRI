import os
import re
import json

input_dir = "../llm-inference/predictions/"
output_dir = os.path.join(input_dir, "formatted-predictions")

os.makedirs(output_dir, exist_ok=True)

def convert_to_json_format(tagged_text, original_text):
    pattern = r'<(HER|VIL|VIC|HER_VIC)>(.*?)</\1>'
    matches = re.finditer(pattern, tagged_text)

    ann = []
    offset = 0
    for match in matches:
        label = match.group(1)
        annotated_text = match.group(2)
        start_index_original = original_text.find(annotated_text, offset)
        end_index_original = start_index_original + len(annotated_text)
        offset = end_index_original

        if label == "HER_VIC":
            labels_list = [["Hero"], ["Victim"]]
        else:
            label_str = "Hero" if label == "HER" else "Villain" if label == "VIL" else "Victim"
            labels_list = [[label_str]]

        for labels in labels_list:
            ann.append({
                "value": {
                    "start": start_index_original,
                    "end": end_index_original,
                    "text": annotated_text,
                    "labels": labels
                }
            })

    return ann

for filename in os.listdir(input_dir):
    if filename.endswith("_predictions.json"):
        model_prompt = filename.replace("_predictions.json", "")
        input_path = os.path.join(input_dir, filename)

        parts = model_prompt.split("_")
        if len(parts) != 2:
            print(f"Skipping unrecognized filename pattern: {filename}")
            continue

        model, prompt_type = parts
        prompt_label = "fewshot" if prompt_type == "few" else "zeroshot" if prompt_type == "zero" else None
        if not prompt_label:
            print(f"Skipping unknown prompt type: {prompt_type}")
            continue

        output_filename = f"{model}_{prompt_label}.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(input_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        final_output = []

        for fname, tagged_text in data.items():
            # remove reasoning output
            if model == "deepseek":
                tagged_text = re.sub(r'<think>.*?</think>[ \n]{0,2}', '', tagged_text, flags=re.DOTALL)

            original_text = re.sub(r'</?(HER|VIL|VIC|HER_VIC)>', '', tagged_text)

            annotations = convert_to_json_format(tagged_text, original_text)
            was_cancelled = False if annotations else True

            final_output.append({
                "data": {"Text": original_text},
                "annotations": [{
                    "result": annotations,
                    "was_cancelled": was_cancelled
                }],
                "filename": fname
            })

        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(final_output, out_file, indent=2, ensure_ascii=False)
