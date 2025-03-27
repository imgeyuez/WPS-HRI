""" 
This file can be used as an inference to detect roles in a given sentence/text.
"""

from transformers import RobertaTokenizerFast, RobertaForTokenClassification
import torch
import json 
from sklearn.preprocessing import LabelEncoder
import numpy as np
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

device = torch.device("cpu")  # Force CPU usage
print(f"Using device: {device}")
m = 'ner_roberta'
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
model_path = f"C:/Users/imgey/Desktop/MASTER_POTSDAM/WiSe2425/PM1_argument_mining/WPS-HRI/roBERTa/{m}"
model = RobertaForTokenClassification.from_pretrained(model_path).to(device)  # Load model to CPU

# Load dataset
with open(r"C:\Users\imgey\Desktop\MASTER_POTSDAM\WiSe2425\PM1_argument_mining\train_dev_test_data\train.json", "r") as file:
    data = json.load(file)

# Extract unique entity labels
all_labels = [ 'O',
    'B-HERO', 'I-HERO',
    'B-VICTIM', 'I-VICTIM',
    'B-VILLAIN', 'I-VILLAIN',
    'B-HERO_VICTIM', 'I-HERO_VICTIM',
    'B-HERO_VILLAIN', 'I-HERO_VILLAIN',
    'B-VICTIM_VILLAIN', 'I-VICTIM_VILLAIN',
    'B-*', 'I-*',
    'B-HERO_HERO', 'I-HERO_HERO'
]


# Convert labels to numbers
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Add special tokens
label_map = {label: i for i, label in enumerate(label_encoder.classes_)}
label_map["PAD"] = -100  # Ignore padding tokens


def predict(tokenized_sentence):
    encoding = tokenizer(tokenized_sentence, is_split_into_words=True, return_tensors="pt").to(device)  # Move to CPU
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    # predictions = torch.argmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]

    index_to_label = {v: k for k, v in label_map.items()}
    decoded_predictions = [index_to_label.get(idx, "UNK") for idx in predictions]
    word_ids = encoding.word_ids()  # Align tokens with original words

    word_preds = []
    current_word = None
    current_label = None

    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue  # Skip special tokens like [CLS], [SEP]

        if word_id != current_word:
            # Start of a new word
            if current_word is not None:
                word_preds.append(current_label)  # Save last word's prediction
            current_word = word_id
            current_label = decoded_predictions[i]  # Take first subword's prediction
        else:
            # Continue word (you can choose to keep first subword's label or majority voting)
            pass

    # Append last word's prediction
    if current_word is not None:
        word_preds.append(current_label)

    # return word_preds

    # filtered_predictions = [
    #     decoded_predictions[i] for i in range(len(decoded_predictions)) if word_ids[i] is not None
    # ]
    # # predictions = [str(label) for label in predictions]
    preds = list()
    for item in word_preds:
        preds.append(item.astype(str).tolist())

    return preds

# print(predict("The Council supported the protection of women and girls who were victims of rape."))
# tokenized_input = ['The', 'Council', 'supported', 'the', 'protection', 'of', 'women', 'and', 'girls', 'who', 'were', 'victims', 'of', 'rape']
# print(predict(tokenized_input))

data_to_evaluate = r'C:\Users\imgey\Desktop\MASTER_POTSDAM\WiSe2425\PM1_argument_mining\WPS-HRI\data\train_dev_test_split\dev.json'
with open(data_to_evaluate, 'r', encoding='UTF-8') as f:
    speeches = json.load(f)

predictions = list()
    
# go through the predictions to evaluate
y_true, y_pred, toks = [], [], []

for speech in speeches:
    filename = speech['filename']
    sentence_predictions = list()
    for sentence in speech['sentences']:
        tokens = sentence['tokens']
        goldlabels = sentence['goldlabels']

        predictions = predict(tokens)

        if len(goldlabels) != len(predictions):
            print(len(goldlabels))
            print(goldlabels)
            print(len(predictions))
            print(predictions)

        toks.append(tokens)
        y_true.append(goldlabels)
        y_pred.append(predictions)

        sent = {'tokens': tokens,
        'goldlabels': goldlabels,
        'preds': predictions}

        sentence_predictions.append(sent)
    predictions.append({
        'filename': filename,
        'sentences': sentence_predictions})
    
for k, (i, j) in enumerate(zip(y_pred, y_true)):
    if len(i) != len(j):
        print(toks[k])

print(f1_score(y_true, y_pred))

print(classification_report(y_true, y_pred))

output_path = f'C:/Users/imgey/Desktop/MASTER_POTSDAM/WiSe2425/PM1_argument_mining/WPS-HRI/RoBERTa/predictions/{m}'
with open(output_path, encoding='UTF-8') as f:
    json.dump(predictions, f, indent=4)



    
