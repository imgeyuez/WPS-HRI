import json
from llm_call import llm_inference, llama_inference
from utils import predictions_json

def load_data(path):
    """
    Function to load the dataset which contains the 
    speeches that are supposed to be annotated.

    Input:
    1. path (str):      os path of the dataset-file.

    Output:
    1. speeches (dict): Dictionary which contains all speeches. Each dictionary 
                        entry represents a speech with the following structure:
                        "filename": str,
                        "sentences": [{tokens: list, goldlabels: list, offset: list}, ...]
    """
    with open(path, "r") as f:
        speeches = json.load(f)

    return speeches

def run_inference(api_key:str, model:str, dataset_path:str):
    """
    Main-Function to run the LLM interface (with deepseek over openrouter).
    Loads the data and iterates over all speeches afterwards to let them 
    annotate by the LLM.

    Input
    1. api_key (str)        : the API key of the user
    2. model (str)          : name of the model they want to use
    3. dataset_path (str)   : path to the dataset

    Output
    None
    .json-file with all predictions per speech
    """

    # initialise empty dictionary to store predictions in for files
    llm_predictions = {}

    # load dataset
    speeches = load_data(dataset_path)  

    # iterate over speeches
    for index, speech in enumerate(speeches):

        # save filename as key in the dictionary 
        filename = speech["filename"]

        # prompt to give the LLM
        prompt= f"""
Task overview
You are given a speech from the United Nations Security Council. Your task is to identify and label characters within the speech. Label each identified character as either Hero, Villain, or Victim. These labels can also be combined to mark that an identified character is portrayed as fulfilling several roles, for example, as Hero and Victim. Some character spans may contain other character spans, for example, when a victim entity is mentioned within a villain entity.

For the annotations, follow these annotation guidelines: 

Annotation guidelines

1. Character Level Rules:
a. Include entire noun phrases (NP): This includes any numerical or descriptive modifiers.
Example: “<HER>60 million Africans</HER> ”
b. Annotate restrictive relative clauses fully.
Example: “<HER>parties that perpetuate acts of violence against women and children</HER> ”
c. Include possessive modifiers within the NP:
Example: “<HER>my delegation</HER> ”
d. When multiple characters are listed together as part of the same NP and share a common role or action, tag the entire sequence as one entity. Do not split or annotate each item separately.
Example: “Allow me to begin by thanking <HER>the Secretary-General, Mr. Kevin Hyland, Mr. Yury Fedotov and Ms. Ilwad Elman</HER> for their briefings.“


2. Role-Definitions:
a. Hero: An entity is labeled as a Hero if it:
helps victims and/or defeats villains
increases agreement within groups or commitment to a cause
recognizes injustice and actively fights to resolve it
has the potential to save others, given their abilities, knowledge, positionality/perspective, but who is not able to do so because of structural discrimination
makes a call to action
calls out or recognizes unjust treatment/violations 

b. Victim: An entity is labeled as a Victim if it:
is weak, good, or innocent and in need of protection
suffers from injustice (e.g., sexual violence, displacement, physical harm)
is excluded from decision-making or denied recognition/power
does not have the same rights as others

c. Villain: An entity is labeled as a Villain if it:
causes anxiety and fear
causes people to lose their daily routines
causes harm or disrupts peace
acts in a way that prevents equal rights and justice for victims
has a negative moral reputation

3. Entities to tag:
You should only tag: 
People 
Organisations  
Countries
Groups  

4. Entities NOT to tag:
Abstract concepts: Do not tag abstract ideas or symbolic references as characters.
Example: “International cooperation”, “French diplomacy”
Entities hoping/welcoming/thanking/commending for something good to happen. These are passive actions.
Laws or treaties: Only annotate United Nations Security Council Resolutions if they are explicitly personified.
Example: In “Resolution 1325 calls for action,” the resolution should be tagged as a hero. In “working towards the implementation of resolution 1325,” the resolution should not be marked.

5. Annotation of character role names: 
Only annotate generic role terms (like “victim”) when no other specification is included.
Example: In “Victims of these atrocious crimes have been waiting for justice”, tag “victims of these atrocious crimes” as Victim.
However, if specific entities are mentioned, like in  “Victims of these atrocious crimes, namely women, have been waiting for justice” only annotate “women” as Victim.

Output Format 
Rewrite the entire speech with in-line tags marking the identified characters. Use these tags:
Hero: <HER>text</HER>
Villain: <VIL>text</VIL>
Victim: <VIC>text</VIC>

Combined roles: 
Hero and Villain: <HER_VIL>text</HER_VIL>
Hero and Victim:  <HER_VIC>text</HER_VIC>
Victim and Villain: <VIC_VIL>text</VIC_VIL>
Hero and Villain and Victim: <HER_VIL_VIC>text</HER_VIL_VIC>

Overlapping annotations should nest tags.

For example: 
“Those who commit crimes against women, including the peacekeeping personnel, should be brought to book.”
In this case, “women” is a victim within the broader entity “those who commit crimes against women, including the peacekeeping personnel”, which would be entirely annotated as villain. 

The output text should be: 
“<VIL>Those who commit crimes against <VIC>women</VIC>, including the peacekeeping personnel</VIL>, should be brought to book.”

In certain cases, characters may be portrayed with multiple roles simultaneously. When this happens, annotate the entity with the combined role. 

For example:
If the input text is: 
“Moreover, in the home, where a woman’s domestic role as spouse and mother is so vital to the well-being of society, her work is always undervalued and underpaid.”

The output text should be: 
“Moreover, in the home, where a <HER_VIC>woman’s</HER_VIC> domestic role as spouse and mother is so vital to the well-being of society, her work is always undervalued and underpaid.”

Explanation: In this case, “woman” is portrayed both as a hero (for her vital role in society) and as a victim (because her work is undervalued). When an entity fits into multiple roles based on the context, use the combined tags.


INSTRUCTION: Rewrite the entire speech with the in-line tags included as specified. Do not provide any additional explanation or formatting. Only include the annotated text—nothing else. The output should strictly be the speech with the appropriate tags for the identified characters.

Here is the speech to annotate: 
“{speech}”
"""
        # call the inference to the model and save its response
        if model == "meta-llama/Llama-3.3-70B-Instruct-Turbo":
            response = llama_inference(model, prompt)
        else:
            response = llm_inference(api_key, model, prompt)

        # add response (predictions) as value to the respective
        # key 
        llm_predictions[filename] = response

    # generate output file with all predictions
    predictions_json(model, llm_predictions)
    
# load API-key 
with open(r"C:\Users\imgey\Desktop\MASTER_POTSDAM\WiSe2425\PM1_argument_mining\WPS\openrouter_api.txt", "r") as f:
    api_key = f.read()

# models that are listed within OpenRouter but do not work
# "meta-llama/llama-3.3-70b-instruct:free"  -> https://github.com/continuedev/continue/issues/3378
models = ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek/deepseek-r1-zero:free"]

# run the script 
for model in models:
    run_inference(api_key, 
                model,
                "./WPS-HRI/data/train_dev_test_split/dev.json")
    print(f"----- Finished with model {model} -----")

