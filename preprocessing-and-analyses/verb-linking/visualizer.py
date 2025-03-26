import streamlit as st
import spacy
from spacy import displacy
import json
import stanza
import re
import graphviz
import textwrap


nlp = spacy.load("en_core_web_sm", exclude=["ner"])
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma,pos,depparse', download_method=stanza.DownloadMethod.REUSE_RESOURCES)

CHARACTER_COLORS = {
    "Hero": "lightblue",
    "Villain": "peru",
    "Victim": "yellow",
}

@st.cache_data
def load_annotations(uploaded_file):
    file_content = uploaded_file.read()
    return json.loads(file_content)

def generate_annotated_html(doc, character_annotations, colors):
    html = "<div style='font-size:0.9em; line-height:4; color:black;'>"
    skip_until = -1  
    i = 0

    while i < len(doc.text):
        if i < skip_until:
            i += 1
            continue

        char_match = next(
            (char for char in character_annotations if i == char["start"]),
            None,
        )

        verb_matches = [
            (char, verb_entry) 
            for char in character_annotations 
            for verb_entry in char.get("verbs", []) 
            if verb_entry["start"] == i
        ]

        if char_match:
            color = colors.get(char_match["labels"][0], "black")
            char_text = doc.text[char_match["start"]:char_match["end"]]
            label = char_match["labels"][0].upper()
            html += f'<span style="background-color:{color}; padding:2px 4px; border-radius:4px; color:black;">{char_text} | <b>{label}</b></span> ' 
            skip_until = char_match["end"]
            i = skip_until
            continue

        if verb_matches:
            verb_start = verb_matches[0][1]["start"]
            verb_end = verb_matches[0][1]["end"]
            verb_text = doc.text[verb_start:verb_end]
            underline_styles = []

            for idx, (char, verb_entry) in enumerate(verb_matches):
                color = colors.get(char["labels"][0], "black")
                offset = 3 + idx * 4
                underline_styles.append(
                    f'text-decoration:underline; text-decoration-color:{color}; '
                    f'text-underline-offset:{offset}px; text-decoration-thickness:2px;'
                )

            styled_text = verb_text
            for style in underline_styles:
                styled_text = f'<span style="{style}">{styled_text}</span>'

            html += styled_text + " "
            skip_until = verb_end
            i = skip_until
            continue

        html += doc.text[i]
        i += 1

    html += "</div>"
    return html

def wrap_text_verbs(text, width=20):
    return "\n".join(textwrap.wrap(text, width)) 

def wrap_text_chars(text, width=20): 
    return "<BR/>".join(textwrap.wrap(text, width))

def generate_dependency_graph(character_annotations):
    if not any(char.get("verbs") for char in character_annotations):
        return None 

    dot = graphviz.Digraph(format="svg")
    dot.attr(rankdir="LR", splines="true", nodesep="0.5")

    added_verbs = {} 
    verb_nodes = set() 
    used_verbs = set() 

    for char in character_annotations:
        for verb_entry in char.get("verbs", []):
            verb = verb_entry["verb"].strip()
            verb_start = verb_entry.get("start", float('inf'))
            verb_end = verb_entry.get("end", float('inf'))
            verb_key = f"{verb.replace(' ', '_')}_{verb_start}_{verb_end}" 

            if verb_key not in added_verbs:
                added_verbs[verb_key] = {
                    "start": verb_start,
                    "end": verb_end,
                    "node": verb_key,
                    "verb": verb
                }
                verb_nodes.add(verb_key)

    character_nodes = {}
    for char in character_annotations:
        for verb_entry in char.get("verbs", []):
            verb = verb_entry["verb"].strip()
            role = verb_entry["role"]
            verb_start = verb_entry.get("start", float('inf'))
            verb_end = verb_entry.get("end", float('inf'))
            verb_key = f"{verb.replace(' ', '_')}_{verb_start}_{verb_end}" 
            character = char["character"]
            character_type = char["labels"][0]  
            span_id = f"{character}_{char['start']}_{char['end']}" 
            color = CHARACTER_COLORS.get(character_type, "gray")

            character_label = f"<B>{wrap_text_chars(character, width=20)}</B>"

            table_label = f"""<
                <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="0">
                    <TR><TD PORT="name" CELLPADDING="6" ALIGN="CENTER" VALIGN="MIDDLE" BGCOLOR="{color}" BORDER="1">{character_label}</TD></TR>
                    <TR><TD CELLPADDING="3" BORDER="0" ALIGN="CENTER" WIDTH="100%"><I>{character_type}</I></TD></TR>
                </TABLE>
            >"""

            character_nodes[span_id] = table_label

            if verb_key in verb_nodes:
                used_verbs.add(verb_key)
                if role == "subject":
                    dot.edge(f"{span_id}:name", verb_key, label=f"<<i>{role}</i>>", fontsize="10", color="black", constraint="true")
                elif role == "advcl":
                    dot.edge(f"{span_id}:name", verb_key, label=f"<<i>{role}</i>>", fontsize="10", color="black", constraint="true", dir="back")
                else:
                    dot.edge(verb_key, f"{span_id}:name", label=f"<<i>{role}</i>>", fontsize="10", color="black", constraint="true")


    for verb_key in used_verbs:
        verb_text = wrap_text_verbs(added_verbs[verb_key]["verb"], width=12)  
        dot.node(
            verb_key, verb_text,
            shape="circle", style="filled",
            fillcolor="white", fontsize="16", fontname="times-bold",
            width="0.9", height="0.8", fixedsize="true"
        )

    for char_id, label in character_nodes.items():
        dot.node(char_id, label=label, shape="plaintext", fontsize="10", width="1.2")

    if used_verbs:
        used_verbs = list(used_verbs)
        used_verbs.sort(key=lambda x: added_verbs[x]["start"], reverse=True)
        dot.body.append(f'{{rank=same; {" ".join(used_verbs)};}}')

    return dot



def visualize_dependency_trees_stanza(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    trees = []
    
    for sentence in sentences:
        if sentence.strip():
            stanza_doc = stanza_nlp(sentence)
            for sent in stanza_doc.sentences:
                displacy_data = {
                    "words": [{"text": word.text, "tag": word.upos} for word in sent.words],
                    "arcs": [
                        {
                            "start": min(word.id - 1, word.head - 1),
                            "end": max(word.id - 1, word.head - 1),
                            "label": word.deprel,
                            "dir": "left" if word.id > word.head else "right",
                        }
                        for word in sent.words if word.head > 0
                    ],
                }
                tree = displacy.render(displacy_data, style="dep", manual=True, page=False)
                trees.append(tree)
    
    return trees

st.title("Dependency Parsing and Character Annotation Visualizer")

st.sidebar.header("Upload Character Annotation Files")
llm_file = st.sidebar.file_uploader("Upload LLM annotations", type=["json"])
manual_file = st.sidebar.file_uploader("Upload manual annotations", type=["json"])

if llm_file and manual_file:
    llm_annotations = load_annotations(llm_file)
    manual_annotations = load_annotations(manual_file)
else:
    st.warning("Please upload both annotation files to proceed.")
    llm_annotations, manual_annotations = None, None

input_text = st.text_area("Enter speech:", "")

if input_text.strip() and (llm_annotations or manual_annotations):
    llm_matched = next(
        (entry for entry in llm_annotations["results"] if input_text.strip() in entry["speech"]),
        None,
    ) if llm_annotations else None

    manual_matched = next(
        (entry for entry in manual_annotations["results"] if input_text.strip() in entry["speech"]),
        None,
    ) if manual_annotations else None

    if llm_matched or manual_matched:
        llm_characters = llm_matched.get("characters_and_verbs", []) if llm_matched else []
        manual_characters = manual_matched.get("characters_and_verbs", []) if manual_matched else []
    else:
        st.warning("No matching speech found in either source.")
        llm_characters, manual_characters = [], []
else:
    llm_characters, manual_characters = [], []

if input_text.strip():
    doc = nlp(input_text)
    colors = {
        "Hero": "lightblue",
        "Villain": "peru",
        "Victim": "yellow",
        "Beneficiary": "plum",
    }

    if llm_characters:
        st.header("LLM Character Annotations")
        llm_html = generate_annotated_html(doc, llm_characters, colors)
        st.markdown(llm_html, unsafe_allow_html=True)

        llm_graph = generate_dependency_graph(llm_characters)
        if llm_graph:
            st.subheader("Verb-Character Graph (LLM Annotations)")
            st.graphviz_chart(llm_graph.source)

    if manual_characters:
        st.header("Manual Character Annotations")
        manual_html = generate_annotated_html(doc, manual_characters, colors)
        st.markdown(manual_html, unsafe_allow_html=True)

        manual_graph = generate_dependency_graph(manual_characters)
        if manual_graph:
            st.subheader("Verb-Character Graph (Manual Annotations)")
            st.graphviz_chart(manual_graph.source)

    st.header("Dependency Tree (Stanza)")
    dep_trees = visualize_dependency_trees_stanza(input_text)
    st.markdown(
        """
        <style>
        .dep-tree-container {
            overflow-x: auto;
            max-width: 100%;
            border: 1px solid black;
            padding: 10px;
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True
        )
    for tree in dep_trees:
        st.markdown(f'<div class="dep-tree-container">{tree}</div>', unsafe_allow_html=True)