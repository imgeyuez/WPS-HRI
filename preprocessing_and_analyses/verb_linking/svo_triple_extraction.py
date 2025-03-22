import json
import csv
from collections import defaultdict

with open('../../data/verb_links_manual_annotations.json', 'r') as f:
    data = json.load(f)

triples = []

for result in data['results']:
    verbs_dict = defaultdict(list)
    
    for character in result['characters_and_verbs']:
        char_name = character['character'].strip()
        char_role = character['labels'][0]
        for verb_info in character['verbs']:
            if not verb_info:
                continue
            verb = verb_info['verb']
            start = verb_info['start']
            end = verb_info['end']
            role = verb_info['role']
            key = (start, end)
            verbs_dict[key].append((char_name, char_role, role, verb))
    
    for key, entries in verbs_dict.items():
        if len(entries) < 2:
            continue  # need at least two characters sharing the verb
        
        # separate entries into subjects and others
        subjects = [entry for entry in entries if entry[2] == 'subject']
        others = [entry for entry in entries if entry[2] != 'subject']
        
        # form triples where a subject is linked to another role through the verb
        for subj in subjects:
            for other in others:
                # ensure the verb text matches
                if subj[3] == other[3]:
                    triples.append({
                        'Character 1': subj[0],
                        'Character Label 1': subj[1],
                        'Dependency Relation 1': subj[2],
                        'Verb': subj[3],
                        'Dependency Relation 2': other[2],
                        'Character Label 2': other[1],
                        'Character 2': other[0]
                    })

with open('../../data/svo_triples_manual_annotations.csv', 'w', newline='') as csvfile:
    fieldnames = ['Character 1', 'Character Label 1', 'Dependency Relation 1', 'Verb', 'Dependency Relation 2', 'Character Label 2', 'Character 2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for triple in triples:
        writer.writerow(triple)