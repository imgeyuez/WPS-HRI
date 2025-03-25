import json
import spacy
import stanza

spacy_nlp = spacy.load("en_core_web_sm", exclude=["ner"])  
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')  

def process_file(file_path, output_path):
    """
    Process a single file to extract character annotations, noun phrases, and linked verbs,
    and calculate match statistics.
    """
    print(f"\n[DEBUG] Processing file: {file_path}")

    with open(file_path, 'r') as file:
        data = json.load(file)

    total_characters = 0
    total_full_matches = 0
    total_partial_matches = 0
    results = []

    for idx, entry in enumerate(data):
        text = entry.get('data', {}).get('Text', "")
        if not text:
            continue

        print(f"\n[DEBUG] Processing paragraph {idx+1}/{len(data)}...")

        # extract character annotations
        character_annotations = extract_character_annotations(entry)
        print(f"[DEBUG] Extracted {len(character_annotations)} character annotations.")

        # extract noun phrases with spaCy
        noun_phrases = extract_noun_phrases_with_positions(text)
        print(f"[DEBUG] Extracted {len(noun_phrases)} noun phrases.")

        # link characters to verbs using Stanza dependency parsing
        linked_verbs = link_characters_to_verbs(text, character_annotations)
        print(f"[DEBUG] Linked verbs to {len(character_annotations)} characters.")

        # check how often character annotations match whole NPs or are contained within them
        full_matches, partial_matches = calculate_match_stats(character_annotations, noun_phrases)

        total_characters += len(character_annotations)
        total_full_matches += full_matches
        total_partial_matches += partial_matches

        results.append({
            "paragraph": text,
            "characters_and_verbs": linked_verbs,
            "noun_phrases": noun_phrases
        })

    # calculate the match rate per dataset
    full_match_rate = total_full_matches / total_characters if total_characters > 0 else 0
    partial_match_rate = total_partial_matches / total_characters if total_characters > 0 else 0

    with open(output_path, 'w') as output_file:
        json.dump({
            "results": results,
            "overall_stats": {
                "total_characters": total_characters,
                "total_full_matches": total_full_matches,
                "total_partial_matches": total_partial_matches,
                "full_match_rate": full_match_rate,
                "partial_match_rate": partial_match_rate
            }
        }, output_file, indent=4)

    print(f"\n[DEBUG] Results saved to {output_path}")
    print(f"Total Characters: {total_characters}")
    print(f"Total Full Matches: {total_full_matches}")
    print(f"Total Partial Matches: {total_partial_matches}")
    print(f"Full Match Rate: {full_match_rate:.2%}")
    print(f"Partial Match Rate: {partial_match_rate:.2%}")


### helper functions ###
def extract_character_annotations(entry):
    """Extract character annotations from the input data."""
    annotations = []
    for annotation in entry.get('annotations', []):
        for result in annotation.get('result', []):
            value = result.get('value', {})
            annotations.append({
                "text": value.get("text", ""),
                "start": value.get("start", 0),
                "end": value.get("end", 0),
                "labels": value.get("labels", [])
            })
    return annotations

def extract_noun_phrases_with_positions(text):
    """
    Extract noun phrases and their positions using spaCy.
    """
    doc = spacy_nlp(text)
    noun_phrases = [
        {
            "text": chunk.text,
            "start": chunk.start_char,
            "end": chunk.end_char
        }
        for chunk in doc.noun_chunks
    ]
    return noun_phrases

def calculate_match_stats(character_annotations, noun_phrases):
    """
    Calculate full and partial match stats between characters and noun phrases.
    """
    full_matches = sum(
        any(
            char['text'] == np['text'] and
            char['start'] == np['start'] and
            char['end'] == np['end']
            for np in noun_phrases
        )
        for char in character_annotations
    )

    partial_matches = sum(
        not any(
            char['text'] == np['text'] and
            char['start'] == np['start'] and
            char['end'] == np['end']
            for np in noun_phrases
        ) and any(
            char['start'] >= np['start'] and
            char['end'] <= np['end']
            for np in noun_phrases
        )
        for char in character_annotations
    )

    return full_matches, partial_matches

###################################
### linking characters to verbs ###
###################################
def create_link_if_valid(verb_links, expanded_verb, main_verb, verb_start, verb_end, role, all_chars):
    """
    Adds the link for (expanded_verb, role) to verb_links if:
      - the verb span does not overlap with (any portion of any) character annotation;
      - expanded_verb has not yet been added to verb_links;
      - the role is not "punct"
    """
    # never create links if dependency relation is punctuation
    if role == "punct":
        print(f"[DEBUG] Skipping link with punct role for verb '{expanded_verb}'")
        return
    
    # never link to verbs within character spans
    for c in all_chars:
        c_s, c_e = c["start"], c["end"]
        if not (verb_end <= c_s or verb_start >= c_e):
            if c_s < verb_end and c_e > verb_start:
                print(f"[DEBUG] Verb '{expanded_verb}' overlaps with character '{c['text']}' - skipping link.")
                return
    
    if expanded_verb in verb_links:
        return
    
    verb_links[expanded_verb] = {
        "role": role,
        "start": verb_start,
        "end": verb_end
    }

def expand_verb_span(verb_token, sentence):
    """
    Expand verb spans to include auxiliaries, particles, negations, and xcomp adjectives.
    """
    components = []
    
    # initialize expanded span boundaries with the main verb token boundaries
    expanded_start = verb_token.start_char
    expanded_end = verb_token.end_char
    components.append({"token": verb_token, "type": "main_verb"})
    
    # collect all direct dependents
    for word in sentence.words:
        if word.head == verb_token.id:
            # auxiliary verbs
            if word.upos == "AUX":
                components.append({"token": word, "type": "aux"})
                expanded_start = min(expanded_start, word.start_char)
                expanded_end = max(expanded_end, word.end_char)
            
            # (basic) negations
            elif word.deprel == "advmod" and word.text.lower() in ["not", "n't"]:
                components.append({"token": word, "type": "neg"})
                expanded_start = min(expanded_start, word.start_char)
                expanded_end = max(expanded_end, word.end_char)
            
            # phrasal verb particles
            elif word.deprel == "compound:prt":
                components.append({"token": word, "type": "prt"})
                expanded_start = min(expanded_start, word.start_char)
                expanded_end = max(expanded_end, word.end_char)
            
            # xcomp adjectives (without their modifiers)
            elif word.deprel == "xcomp" and word.upos == "ADJ":
                components.append({"token": word, "type": "xcomp_adj"})
                expanded_start = min(expanded_start, word.start_char)
                expanded_end = max(expanded_end, word.end_char)
            
            # xcomp verbs with adjective complements
            elif word.deprel == "xcomp" and word.upos == "VERB":
                # include adjectives that are direct dependents of xcomp verbs
                for inner_word in sentence.words:
                    if inner_word.head == word.id and inner_word.upos == "ADJ":
                        # include only the adjective itself, not its modifiers
                        components.append({"token": inner_word, "type": "xcomp_adj"})
                        expanded_start = min(expanded_start, inner_word.start_char)
                        expanded_end = max(expanded_end, inner_word.end_char)
    
    # sort characters by their position in the text
    components.sort(key=lambda x: x["token"].start_char)
    component_texts = [comp["token"].text for comp in components]
    
    # combine all parts
    expanded_verb = " ".join(component_texts)
    
    # fix spacing around negations
    if "n't" in expanded_verb:
        expanded_verb = expanded_verb.replace(" n't", "n't")
    
    main_verb = verb_token.text
    return expanded_verb, main_verb, expanded_start, expanded_end

def is_blocking_dependency_between(verb, conjunct, sentence):
    """
    Check if a direct object ('obj') or a prepositional phrase ('obl')
    intervenes between a verb and its conjunct. Unlink verb and conjunct if so.
    """
    if not verb or not conjunct:
        return False
    verb_pos = verb.id
    conjunct_pos = conjunct.id
    for w in sentence.words:
        if verb_pos < w.id < conjunct_pos and w.deprel in {"obj", "obl"}:
            return True
    return False


def find_referent(word, sentence):
    """Find the closest noun before a relative pronoun ("that", "which", "who")."""
    candidates = [w for w in sentence.words if w.upos in {"NOUN", "PROPN"} and w.id < word.id]
    return candidates[-1] if candidates else None


def link_conjunct_characters(noun_token, verb_links, role, all_chars, sentence):
    """
    Ensures characters linked via the 'conj' dependency relation to other characters
    inherit all their verb links.
    
    - Finds characters linked via 'conj'.
    - Copies verb links from the main character to its conjuncts.
    """
    stack = [noun_token]
    visited = set()
    while stack:
        current = stack.pop()
        visited.add(current.id)

        # link all siblings that are conj to current noun
        for w in sentence.words:
            if w.id not in visited:
                if (w.head == current.id and w.deprel == "conj" and w.upos in {"NOUN", "PROPN", "ADJ"}):
                    stack.append(w)
                if (current.head == w.id and current.deprel == "conj" and w.upos in {"NOUN", "PROPN", "ADJ"}):
                    stack.append(w)
    return visited


def link_characters_to_verbs(text, character_annotations):
    """
    Link characters to their governing and modifying verbs or nominal predicates using Stanza dependency parsing.
    """
    
    print("\n[DEBUG] Running Stanza dependency parsing...")
    doc = stanza_nlp(text)
    linked_results = []
    character_verb_mappings = {}

    char_by_start = {c['start']: c for c in character_annotations}

    for char in character_annotations:
        char_tokens = []
        char_sentence = None

        # find all the tokens that make up the character
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.start_char is not None and word.end_char is not None:
                    if (char["start"] <= word.start_char < char["end"]):
                        char_tokens.append(word)
                        char_sentence = sentence

        if not char_tokens or not char_sentence:
            linked_results.append({
                "character": char["text"],
                "start": char["start"],
                "end": char["end"],
                "labels": char["labels"],
                "verbs": [],
            })
            continue

        verb_links = {}
        
        ##################################################################
        # 1) for each token in char_tokens:
        #    - any verbs that govern it (via nsubj, obj, etc.)
        #    - any nominal heads that have a copula (e.g. "is source")
        #    - multi-level nmod or conj expansions
        ##################################################################
        visited = set()
        tokens_to_visit = char_tokens[:]

        while tokens_to_visit:
            token = tokens_to_visit.pop()
            if token.id in visited:
                continue
            visited.add(token.id)

            if token.head > 0:
                head = next((w for w in char_sentence.words if w.id == token.head), None)
            else:
                head = None

            deprel = token.deprel

            ### if a head is a verb or auxiliary verb, check for other particles or auxiliaries to link to verb ###
            if head and head.upos in {"VERB", "AUX"}:
                expanded_verb, main_verb, vs, ve = expand_verb_span(head, char_sentence)
                # and determine role
                role_map = {
                    "nsubj": "subject", "nsubj:pass": "subject", "csubj": "subject",
                    "obj": "object", "iobj": "object",
                    "obl": "oblique", "advmod": "modifier", "amod": "modifier"
                }
                role = role_map.get(deprel, deprel)

                if role:
                    # all characters that are conjuncts of the current character inherit verb links
                    conj_chain = link_conjunct_characters(token, verb_links, role, character_annotations, char_sentence)
                    for conj_id in conj_chain:
                        conj_token = next((x for x in char_sentence.words if x.id == conj_id), None)
                        if conj_token is not None:
                            create_link_if_valid(verb_links, expanded_verb, main_verb, vs, ve, role, character_annotations)

            # if the head of the character is a NOUN/PROPN/ADJ, check for the copula/whether it's a nominal modifier
            if head and head.upos in {"NOUN", "PROPN", "ADJ"}:
                role_map = {
                    "nsubj": "subject", "nsubj:pass": "subject", "csubj": "subject",
                    "obj": "object", "iobj": "object",
                    "obl": "oblique", "advmod": "modifier", "amod": "modifier",
                    "nmod": "nmod"
                }
                role = role_map.get(deprel, None)

                cop_word = next((w for w in char_sentence.words if w.head == head.id and w.deprel == "cop"), None)
                if role in {"subject", "object"} and cop_word:
                    # expand copula to include auxiliaries, negations, etc.
                    expanded_cop, main_cop, cs, ce = expand_verb_span(cop_word, char_sentence)
                    combined_pred = f"{expanded_cop} {head.text}".strip()
                    combined_start = min(cs, head.start_char)
                    combined_end = max(ce, head.end_char)
                    create_link_if_valid(verb_links, combined_pred, main_cop, combined_start, combined_end, role, character_annotations)

                if role == "nmod":
                    tokens_to_visit.append(head) 

                if deprel == "conj":
                    tokens_to_visit.append(head)

                if role in {"subject", "object"}:
                    tokens_to_visit.append(head)

        ##################################################################
        # 2) Now link child verbs that modify these heads (advcl, xcomp, conj, acl, etc.)
        #    (i.e., scan any verbs that have a head in the set of
        #    verb tokens linked above)
        ##################################################################

        verb_token_mapping = {}
        for verb_text, info in verb_links.items():
            # find the main verb tokens in the expanded verb span
            for word in char_sentence.words:
                # check if this word is part of any expanded verb span
                if word.start_char >= info["start"] and word.end_char <= info["end"] and word.upos in {"VERB", "AUX"}:
                    verb_token_mapping[word.id] = {"text": verb_text, "role": info["role"]}

        ### verbs that modify the governing verb ###
        for verb_id, verb_info in verb_token_mapping.items():
            parent_verb_text = verb_info["text"]
            parent_role = verb_info["role"]
            parent_verb = next((w for w in char_sentence.words if w.id == verb_id), None)
            
            if not parent_verb:
                continue
                
            print(f"[DEBUG] Looking for modifiers of '{parent_verb.text}' (part of '{parent_verb_text}')")
            
            for word in char_sentence.words:
                if word.head == parent_verb.id and word.upos == "VERB":
                    # 'acl', 'advcl', 'acl:relcl' (modifiers of the governing verb)
                    if word.deprel in {"acl", "advcl", "acl:relcl"}:
                        expanded, mainv, s0, s1 = expand_verb_span(word, char_sentence)
                        print(f"[DEBUG] Found '{word.deprel}' modifying verb '{parent_verb_text}': '{expanded}'.")
                        
                        # find the subject or object of the relative clause
                        relcl_subject = next((w2 for w2 in char_sentence.words 
                                              if w2.head == word.id and w2.deprel in {"nsubj","nsubj:pass","csubj"}), None)
                        relcl_object = next((w2 for w2 in char_sentence.words 
                                             if w2.head == word.id and w2.deprel in {"obj","iobj"}), None)
                        role = word.deprel
                        
                        if relcl_subject and relcl_subject.text.lower() in {"who", "that", "which"}:
                            refer = find_referent(relcl_subject, char_sentence)
                            if refer and (char["start"] <= refer.start_char < char["end"]):
                                role = "subject"
                        elif relcl_subject:
                            if char["start"] <= relcl_subject.start_char < char["end"]:
                                role = "subject"
                        
                        if relcl_object and relcl_object.text.lower() in {"who", "that", "which"}:
                            refer = find_referent(relcl_object, char_sentence)
                            if refer and (char["start"] <= refer.start_char < char["end"]):
                                role = "object"
                        elif relcl_object:
                            if char["start"] <= relcl_object.start_char < char["end"]:
                                role = "object"
                                
                        create_link_if_valid(verb_links, expanded, mainv, s0, s1, role, character_annotations)
                    
                    # 'xcomp' and 'conj' should inherit role from parent verb
                    elif word.deprel in {"xcomp", "conj"}:
                        if word.deprel == "conj":
                            if is_blocking_dependency_between(parent_verb, word, char_sentence):
                                print(f"[DEBUG] Blocking conjunct '{word.text}' from inheriting '{parent_verb_text}' due to an intervening blocker.")
                                continue
                        
                        expanded, mainv, s0, s1 = expand_verb_span(word, char_sentence)
                        print(f"[DEBUG] Found '{word.deprel}' '{expanded}' modifying '{parent_verb_text}'. Inheriting role '{parent_role}'.")
                        create_link_if_valid(verb_links, expanded, mainv, s0, s1, parent_role, character_annotations)
                        
                        # for xcomps, also check for nested conjuncts
                        if word.deprel == "xcomp":
                            # find all conjuncts of this xcomp
                            xcomp_conjuncts = [w for w in char_sentence.words 
                                              if w.head == word.id and 
                                              w.upos == "VERB" and 
                                              w.deprel == "conj"]
                            
                            for conj_verb in xcomp_conjuncts:
                                if not is_blocking_dependency_between(word, conj_verb, char_sentence):
                                    conj_expanded, conj_mainv, conj_s0, conj_s1 = expand_verb_span(conj_verb, char_sentence)
                                    print(f"[DEBUG] Found nested conjunct '{conj_expanded}' of xcomp '{expanded}'. Inheriting role '{parent_role}'.")
                                    create_link_if_valid(verb_links, conj_expanded, conj_mainv, conj_s0, conj_s1, parent_role, character_annotations)

        # handle acl modifiers of nouns
        for word in char_sentence.words:
            if word.upos == "VERB" and word.deprel in {"acl", "acl:relcl"}:
                # if the head is a noun
                head_noun = next((w for w in char_sentence.words if w.id == word.head), None)
                if head_noun:
                    # check if any linked verbs contain this noun
                    for vtext, info in list(verb_links.items()):
                        if head_noun.text in vtext.split():
                            expanded, mainv, s0, s1 = expand_verb_span(word, char_sentence)
                            role = word.deprel
                            # check for subject/object override
                            relcl_subject = next((w2 for w2 in char_sentence.words 
                                                  if w2.head == word.id and w2.deprel in {"nsubj", "nsubj:pass","csubj"}), None)
                            relcl_object = next((w2 for w2 in char_sentence.words 
                                                 if w2.head == word.id and w2.deprel in {"obj", "iobj"}), None)
                            if relcl_subject and char["start"] <= relcl_subject.start_char < char["end"]:
                                role = "subject"
                            if relcl_object and char["start"] <= relcl_object.start_char < char["end"]:
                                role = "object"
                                
                            create_link_if_valid(verb_links, expanded, mainv, s0, s1, role, character_annotations)
                            
                            # check for xcomps of this acl
                            acl_xcomps = [w for w in char_sentence.words 
                                         if w.head == word.id and 
                                         w.upos == "VERB" and 
                                         w.deprel == "xcomp"]
                            
                            for xcomp_verb in acl_xcomps:
                                xcomp_expanded, xcomp_mainv, xcomp_s0, xcomp_s1 = expand_verb_span(xcomp_verb, char_sentence)
                                create_link_if_valid(verb_links, xcomp_expanded, xcomp_mainv, xcomp_s0, xcomp_s1, role, character_annotations)

                                xcomp_conjuncts = [w for w in char_sentence.words 
                                                  if w.head == xcomp_verb.id and 
                                                  w.upos == "VERB" and 
                                                  w.deprel == "conj"]
                                
                                for conj_verb in xcomp_conjuncts:
                                    if not is_blocking_dependency_between(xcomp_verb, conj_verb, char_sentence):
                                        conj_expanded, conj_mainv, conj_s0, conj_s1 = expand_verb_span(conj_verb, char_sentence)
                                        create_link_if_valid(verb_links, conj_expanded, conj_mainv, conj_s0, conj_s1, role, character_annotations)

        character_verb_mappings[char["start"]] = verb_links

    # final results
    for char in character_annotations:
        final_verb_links = character_verb_mappings.get(char["start"], {})
        linked_results.append({
            "character": char["text"],
            "start": char["start"],
            "end": char["end"],
            "labels": char["labels"],
            "verbs": [
                {
                    "verb": v,
                    "role": info["role"],
                    "start": info["start"],
                    "end": info["end"]
                }
                for v, info in final_verb_links.items()
            ]
        })

    return linked_results

# don't forget to add LLM and RoBERTa files!!
file_paths = [
    '../../data/manual_character_annotations.json',
    '../../data/llm_annotations/deepseek_zeroshot.json'
]
output_paths = [
    '../../data/verb_links_manual_annotations.json',
    '../../data/verb_links_deepseek_zeroshot_annotations.json'
]

for file_path, output_path in zip(file_paths, output_paths):
    process_file(file_path, output_path)