import json
import torch
from transformers import RobertaTokenizerFast
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaForTokenClassification, AdamW
from torch.optim import SGD
# from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report


def encode_example(sentence):
    tokens = sentence["tokens"]
    labels = sentence["goldlabels"]

    # Tokenize
    encoding = tokenizer(tokens, is_split_into_words=True, truncation=True, padding="max_length", max_length=128)

    # Align labels with tokens
    encoded_labels = []
    word_ids = encoding.word_ids()  # Map subword tokens to word indices

    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            encoded_labels.append(label_map["PAD"])  # Padding token
        elif word_idx != prev_word_idx:
            encoded_labels.append(label_map[labels[word_idx]])  # First sub-token
        else:
            encoded_labels.append(label_map[labels[word_idx]])  # Inside subword (use same label)
        prev_word_idx = word_idx

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": encoded_labels
    }



def evaluate(model, dataloader, tokenizer, device, label_map, model_name):
    model.eval()  # Set model to evaluation mode

    results_list = []  # Store results per sentence

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Convert label map to an index-to-label dictionary
            index_to_label = {v: k for k, v in label_map.items()}  # Reverse mapping

            # Decode tokens
            decoded_tokens = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            for i in range(len(input_ids)):  # Iterate over each sentence in batch
                # Mask to remove padding
                mask = attention_mask[i] == 1  # True for actual tokens, False for padding tokens

                # Get tokens, predictions, and gold labels without padding
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i][mask])
                pred_labels = [index_to_label.get(idx.item(), "UNK") for idx in predictions[i][mask]]
                gold_labels = [index_to_label.get(idx.item(), "UNK") for idx in labels[i][mask]]

                # Append structured data to results list
                results_list.append({
                    "tokens": tokens,
                    "gold_labels": gold_labels,
                    "predictions": pred_labels
                })

    # Compute overall evaluation metrics
    all_gold_labels = [idx for r in results_list for idx in r["gold_labels"]]
    all_pred_labels = [idx for r in results_list for idx in r["predictions"]]

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_gold_labels, all_pred_labels, average="macro", zero_division=0
    )

    # Store overall metrics in the JSON output
    final_results = {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions_per_sentence": results_list  # Store structured sentence-wise results
    }

    # Save results to a JSON file
    save_path = f"./predictions/{model_name}_predictions.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"\nEvaluation results saved to {save_path}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    return precision, recall, f1



# Load the tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

# Load train, dev and test datasets
with open("../data/train_dev_test_split/train.json", "r") as file:
    traindata = json.load(file)
with open("../data/train_dev_test_split/dev.json", "r") as file:
    devdata = json.load(file)
# with open(r".\data\train_dev_test_split\test.json", "r") as file:
#     testdata = json.load(file)
  
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

# Process data

# Convert entire dataset
train_dataset = [encode_example(sentence) for speech in traindata for sentence in speech["sentences"]]
dev_dataset = [encode_example(sentence) for speech in devdata for sentence in speech["sentences"]]
# test_dataset = [encode_example(sentence) for speech in testdata for sentence in speech["sentences"]]

# Save processed dataset
torch.save(train_dataset, "./datasets/train_dataset.pt")
torch.save(dev_dataset, "./datasets/dev_dataset.pt")
# torch.save(test_dataset, "test_dataset.pt")

from dataloader import train_loader, dev_loader

num_labels = len(label_map) - 1  # Exclude "PAD" label
model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=num_labels)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

#################
# HYPERPARAMETER#
#################

# learning rate, epochs, optimizer
hyperparameters = {
    "m0": [5e-5, 1, "AdamW"],
    "m1": [5e-5, 3, "AdamW"],
    "m2": [5e-5, 30, "AdamW"],
    "m3": [1e-5, 3, "AdamW"],
    "m4": [1e-5, 30, "AdamW"],
    "m5": [5e-5, 30, "SGD"],
    "m6": [3e-5, 30, "SGD"],
}

for key, values in hyperparameters.items():

    lr, epochs, optimizer_type = values

    if optimizer_type == "AdamW":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif optimizer_type == "SGD":
        optimizer = SGD(model.parameters(), lr=lr)

    model.train()

    # Training loop

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        total_loss = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Loss: {total_loss / len(train_loader)}")

    # Save model
    save_path = f"./models/ner_roberta_{key}"
    model.save_pretrained(save_path)

