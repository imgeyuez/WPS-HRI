import re
import json

input_path = "../data/llm_annotations/deepseek_zeroshot.txt"
output_path = "../data/llm_annotations/deepseek_zeroshot.json"

with open(input_path, "r", encoding="utf-8") as file:
    tagged_text = file.read().replace("\\n", " ")
    tagged_text = tagged_text.replace('\\"', '"')  


 

# Step 2: Create the original (untagged) version of the text
original_text = re.sub(r'</?(HER|VIL|VIC|HER_VIC)>', '', tagged_text)

# Step 3: Convert inline annotations to structured format
def convert_to_json_format(text_to_parse, original_text):
    pattern = r'<(HER|VIL|VIC|HER_VIC)>(.*?)</\1>'
    matches = re.finditer(pattern, text_to_parse)

    ann = []
    offset = 0
    for match in matches:
        label = match.group(1)
        annotated_text = match.group(2)
        start_index_tagged = match.start(2)

        start_index_original = original_text.find(annotated_text, offset)
        end_index_original = start_index_original + len(annotated_text)

        offset = end_index_original 

        if label == "HER_VIC":
            labels = ["Hero", "Victim"]
        else:
            labels = ["Hero" if label == "HER" else "Villain" if label == "VIL" else "Victim"]

        ann.append({
            "value": {
                "start": start_index_original,
                "end": end_index_original,
                "text": annotated_text,
                "labels": labels
            }
        })

    return ann

# Step 4: Generate final formatted output
annotations = convert_to_json_format(tagged_text, original_text)
was_cancelled = False if annotations else True

final_output = [{
    "data": {
        "Text": original_text
    },
    "annotations": [
        {
            "result": annotations,
            "was_cancelled": was_cancelled
        }
    ]
}]

# Step 5: Save to JSON file
with open(output_path, "w", encoding="utf-8") as out_file:
    json.dump(final_output, out_file, indent=2, ensure_ascii=False)

output_path
