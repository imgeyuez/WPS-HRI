import json
from llm_call import llm_inference


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

def run_inference(api_key:str, models:list, prompt_method:str, dataset_path:str):
    """
    Main-Function to run the LLM interface (with deepseek over openrouter).
    Loads the data and iterates over all speeches afterwards to let them 
    annotate by the LLM.

    Input
    1. api_key (str)        : the API key of the user
    2. model (str)          : name of the model they want to use
    3. prompt_method (str)  : if the llm gets zero-shot or few-shot prompt.
                                Input is either 'few' or 'zero'.
    3. dataset_path (str)   : path to the dataset

    Output
    None
    .json-file with all predictions per speech for each model
    """

    # load dataset
    speeches = load_data(dataset_path)  

    # prompts to give the LLM
    if prompt_method == "zero":
        # prompt for zero-shot
        prompt= """
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
{}
        """
            
    else:
        # prompt for few-shot
        prompt= """
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

Below is an annotated example of how to identify and label characters within a speech from the United Nations Security Council. Use this format to annotate the speech provided.

Annotated Example: 
“Allow me to begin by thanking <HER>the Secretary-General, Mr. Kevin Hyland, Mr. Yury Fedotov and Ms. Ilwad Elman</HER> for their briefings.  France associates itself with the statement to be delivered by the observer of the European Union.  The actions committed by <VIL>Da'esh</VIL> in the Middle East and by <VIL>Boko Haram</VIL> in Africa are a dramatic illustration of the links that exist today between threats to international peace and security and human trafficking. At the global level, trafficking in <VIC>persons</VIC> and trafficking in drugs and counterfeit currency are among the most profitable. Their annual profits are estimated at $32 billion. It is one of the most extensive forms of trafficking. Sexual exploitation, forced labour, servitude, kidnapping for forced prostitution, rape - the list of atrocities committed in armed conflict is, unfortunately, long. Deriving profits from <VIC>human beings</VIC> and considering <VIC>them</VIC> as merchandise, <VIL>the traffickers, as well as consumers and users and the clients of sexual exploitation</VIL>, clearly and brutally violate their <VIL>victims'</VIL> human rights and further stoke the causes of conflict.  <HER>The international community</HER> has invested heavily in this problem since the beginning of this century, but further efforts are needed in order to address the scourge of human trafficking. I therefore welcome the initiative of <HER>the United Kingdom</HER> during its presidency of the Security Council and thank <HER>that country</HER> for giving us this opportunity to have an exchange on this very important subject.  For <HER>France</HER>, the issue oftrafficking in <VIC>human beings</VIC> and slavery, especially that of <VIC>women and children</VIC>, is of major importance. The statistics are, unfortunately, well known, but we must constantly point them out: 80 per cent of the victims of trafficking are <VIC>women and children</VIC>. The challenges are also well known: the identification of <VIC>victims</VIC> is still in its infancy, and organized mechanisms for fighting this scourge vary greatly between countries. Despite progress since the entry into force of the Palermo Protocol to Prevent, Suppress and Punish Trafficking in Persons, Especially Women and Children, still too few prosecutions have been initiated in cases involving the crime of human trafficking. <VIC>The victims</VIC> themselves do not always assert their rights and very often are insufficiently protected. In that context, international cooperation must be stepped up so as to increase the geographic coverage of the legislation providing effective protection against <VIL>networks</VIL> and to improve international cooperation aimed at dismantling <VIL>those networks</VIL>. Prevention, protection and the fight against impunity are the three priorities of French diplomacy in the fight against trafficking in <VIC>human beings</VIC>.  Since human trafficking is now an integral part of the strategy of <VIL>certain terrorist groups</VIL> and it fuels transnational organized crime, the Security Council has a special responsibility in combating this scourge. The adoption of resolution 2331 (2016), last December, at the initiative of <HER>Spain</HER>, was a major step forward towards better addressing the link between trafficking in <VIC>human beings</VIC>, sexual violence and terrorism. France very much looks forward to the report to be prepared by the Secretary-General by the end of the year. <HER>We</HER> have in place a robust international legal framework and appropriate tools, in particular the United Nations Convention against Transnational Organized Crime and its Protocols and <HER>the United Nations Office on Drugs and Crime</HER>, which is doing sterling work in this field.  On International Women's Rights Day, <HER>the President of the French Republic</HER> also announced that <HER>France</HER> would propose an additional protocol to the Convention for the Elimination of All Forms of Discrimination against Women. That protocol would address violence against <VIC>women</VIC> in order to complement the existing international framework. But we must ensure that the obligations arising from that legal framework are effectively implemented. Our words must now be translated into action.  Rest assured, Mr. President, that <HER>France</HER> will continue to play its full part in those efforts.”

INSTRUCTION: Rewrite the entire speech with the in-line tags included as specified. Do not provide any additional explanation or formatting. Only include the annotated text—nothing else. The output should strictly be the speech with the appropriate tags for the identified characters.

Here is the speech to annotate: 
{}
        """
            

    # iterate over all models
    for model in models:

        # initialise empty dictionary to store predictions in for files
        llm_predictions = {}

        # iterate over speeches
        for speech in speeches:

            # save filename as key in the dictionary 
            filename = speech["filename"]
            
            with open("C:/Users/imgey/Desktop/MASTER_POTSDAM/WiSe2425/PM1_argument_mining/WPS-HRI/data/manual_character_annotations.json", "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # fetch the respective file from
            # manual_character_annotations.json
            for entry in json_data:
                cleaned_name = entry["filename"].split("/")[0]
                if cleaned_name == filename:
                    speech_text = entry["data"]["Text"]
                    break
            
            # include the speech into the prompt
            llm_prompt = prompt.format(speech_text)
            
            # call the inference to the model and save its response
            response = llm_inference(api_key,
                                     model, 
                                     llm_prompt)

            # save response in dict
            llm_predictions[filename] = response

        # generate output file with all predictions
        predictions_json(model, prompt_method, llm_predictions)
    
def predictions_json(model, prompt_method, predictions:dict):
    """
    This function takes in a dictionary with predictions
    of a model and stores them within a .json-file.

    Input:
    1. model (str)          : name of used llm.
    2. prompt_method (str)  : used prompt-method. 'few' or 'zero'
    3. predictions (dict)   : dictionary with all responses of llm.
    """

    if "deepseek" in model:
        model_name = "deepseek"
    elif "llama" in model:
        model_name = "llama"
    # define outputfile path
    output_filename = f"./WPS-HRI/predictions/{model_name}_{prompt_method}_predictions.json"

    with open(output_filename, "w") as f:    
        json.dump(predictions, f, indent=4)


######################################################################################################################
# load API-key 
# Read the API key from the file
with open(r"C:\Users\imgey\Desktop\MASTER_POTSDAM\WiSe2425\PM1_argument_mining\WPS\togetherai_api.txt", "r") as f:
    api_key = f.read().strip()  # Make sure there are no extra spaces or newline characters

# list of models to use (on together ai)
models = ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-R1"] 

# run the script & specify which prompt to use
run_inference(api_key, 
            models,
            "zero", # alternative: "few"
            "./WPS-HRI/data/train_dev_test_split/dev.json")
    

