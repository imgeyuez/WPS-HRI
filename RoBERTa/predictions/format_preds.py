import json

with open(r"C:\Users\imgey\Desktop\MASTER_POTSDAM\WiSe2425\PM1_argument_mining\WPS-HRI\RoBERTa\predictions\finetuned_ner_roberta_m4.json", "r", encoding="UTF-8") as file:
    preds = json.load(file)

for speech in preds:
    print(speech["filename"])

    for sentence in speech["sentences"]:
        for i, token in enumerate(sentence["tokens"]):
            print(token, "\t", sentence["preds"][i], "\t", sentence["goldlabels"][i])
        print("\n")
        # print(" ".join(sentence["tokens"]))
        # print(" ".join(sentence["preds"]))
        # print(" ".join(sentence["goldlabels"]))
        # print("\n")
