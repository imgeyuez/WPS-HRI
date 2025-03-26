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
        prompt= """Your goal: Rewrite the input speech with only in-line character role annotations, using precise tags, leaving spacing and spelling as is, and following all structural and role-based rules.

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
e. Do not include punctuation at the end of annotations.
Example: “<VIC>men and boys</VIC>.”
f. Do not annotate predicate nominatives or any other descriptions of characters.
Example: “<VIC>Women</VIC> are not only victims.” Here, victims should not be annotated.
Example: “<HER>women</HER> as  agents of peace”
g. Do not annotate positions or groups when they are only mentioned in the abstract, without references to the achievements of the particular entity.
Example: “the establishment of the post of the Special Representative of the Secretary General on sexual violence in situations of armed conflict” 


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
a. People 
b. Organisations  
c. Countries
d. Groups  
e. UN Resolutions. Only annotate United Nations Resolutions if they are explicitly personified.
Example: In “Resolution 1325 calls for action,” the resolution should be tagged as a hero. In “working towards the implementation of resolution 1325,” the resolution should not be marked.

4. Entities NOT to tag:
a. Abstract concepts: Do not tag abstract ideas or symbolic references as characters.
Examples: “International cooperation,” “sexual violence,” “<HER>women’s</HER> participation.” In the latter example, women are the entities that are taking heroic action, while their participation refers to a concept.
b. Entities hoping for/welcoming/thanking/commending something good. These are passive actions.
c. Laws or Treaties. 
d. Speakers at the beginning of the input, separated from the content of the speech with a colon.

5. Annotation of character role labels: 
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

Example: 
“Those who commit crimes against women, including the peacekeeping personnel, should be brought to book.”
In this case, “women” is a victim within the broader entity “those who commit crimes against women, including the peacekeeping personnel”, which would be entirely annotated as villain. 

The output text should be: 
“<VIL>Those who commit crimes against <VIC>women</VIC>, including the peacekeeping personnel</VIL>, should be brought to book.”

In certain cases, characters may be portrayed with multiple roles simultaneously. When this happens, annotate the entity with the combined role. Furthermore, the same entity can take on different roles throughout the speech. For instance, while “women” might be classified as <VIC>women</VIC> in one sentence, the same entity can also be classified as <HER>women</HER> in a different part of the speech.

Example:
If the input text is: 
“The equal right to decision-making and participation, along with women's empowerment, is crucial to ensure a functioning society and peace and justice in the aftermath of conflicts.”

The output text should be: 
“The equal right to decision-making and participation, along with <HER_VIC>women’s</HER_VIC> empowerment, is crucial to ensure a functioning society and peace and justice in the aftermath of conflicts.”


Explanation: In this case, “women” are portrayed both as victims (since they need external help to be empowered) as well as heroes (since they contribute to a peaceful society). When an entity fits into multiple roles based on the context, use the combined tags.

INSTRUCTION: Rewrite the entire speech with the in-line tags included as specified. When annotating, do not rely on any external world knowledge about the entities described in the text. Focus on the way that the speaker portrays the entities in the text. In your output, do not provide any additional explanation or formatting. Make sure to include all spaces from the original text. Multiple spaces are not equivalent to line breaks. Keep the exact same number of spaces between words. Do not replace spaces with new lines. Do not change capitalization, punctuation, or spelling even if you find mistakes. Only include the annotated text—nothing else. The output should strictly be the speech with the appropriate tags for the identified characters.

Annotate this speech: 
{}"""
            
    else:
        # prompt for few-shot
        prompt= """Your goal: Rewrite the input speech with only in-line character role annotations, using precise tags, leaving spacing and spelling as is, and following all structural and role-based rules.

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
e. Do not include punctuation at the end of annotations.
Example: “<VIC>men and boys</VIC>.”
f. Do not annotate predicate nominatives or any other descriptions of characters.
Example: “<VIC>Women</VIC> are not only victims.” Here, victims should not be annotated.
Example: “<HER>women</HER> as  agents of peace”
g. Do not annotate positions or groups when they are only mentioned in the abstract, without references to the achievements of the particular entity.
Example: “the establishment of the post of the Special Representative of the Secretary General on sexual violence in situations of armed conflict” 


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
a. People 
b. Organisations  
c. Countries
d. Groups  
e. UN Resolutions. Only annotate United Nations Resolutions if they are explicitly personified.
Example: In “Resolution 1325 calls for action,” the resolution should be tagged as a hero. In “working towards the implementation of resolution 1325,” the resolution should not be marked.

4. Entities NOT to tag:
a. Abstract concepts: Do not tag abstract ideas or symbolic references as characters.
Examples: “International cooperation,” “sexual violence,” “<HER>women’s</HER> participation.” In the latter example, women are the entities that are taking heroic action, while their participation refers to a concept.
b. Entities hoping for/welcoming/thanking/commending something good. These are passive actions.
c. Laws or Treaties. 
d. Speakers at the beginning of the input, separated from the content of the speech with a colon.

5. Annotation of character role labels: 
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

Example: 
“Those who commit crimes against women, including the peacekeeping personnel, should be brought to book.”
In this case, “women” is a victim within the broader entity “those who commit crimes against women, including the peacekeeping personnel”, which would be entirely annotated as villain. 

The output text should be: 
“<VIL>Those who commit crimes against <VIC>women</VIC>, including the peacekeeping personnel</VIL>, should be brought to book.”

In certain cases, characters may be portrayed with multiple roles simultaneously. When this happens, annotate the entity with the combined role. Furthermore, the same entity can take on different roles throughout the speech. For instance, while “women” might be classified as <VIC>women</VIC> in one sentence, the same entity can also be classified as <HER>women</HER> in a different part of the speech.

Example:
If the input text is: 
“The equal right to decision-making and participation, along with women's empowerment, is crucial to ensure a functioning society and peace and justice in the aftermath of conflicts.”

The output text should be: 
“The equal right to decision-making and participation, along with <HER_VIC>women’s</HER_VIC> empowerment, is crucial to ensure a functioning society and peace and justice in the aftermath of conflicts.”


Explanation: In this case, “women” are portrayed both as victims (since they need external help to be empowered) as well as heroes (since they contribute to a peaceful society). When an entity fits into multiple roles based on the context, use the combined tags.

Below is an annotated example of how to identify and label characters within a speech from the United Nations Security Council. Use this format to annotate the speech provided.

Annotated Example 1: 
Ms. Rossignol (France) (spoke in French): “Allow me to begin by thanking <HER>the Secretary-General, Mr. Kevin Hyland, Mr. Yury Fedotov and Ms. Ilwad Elman</HER> for their briefings.  France associates itself with the statement to be delivered by the observer of the European Union.  The actions committed by <VIL>Da'esh</VIL> in the Middle East and by <VIL>Boko Haram</VIL> in Africa are a dramatic illustration of the links that exist today between threats to international peace and security and human trafficking. At the global level, trafficking in <VIC>persons</VIC> and trafficking in drugs and counterfeit currency are among the most profitable. Their annual profits are estimated at $32 billion. It is one of the most extensive forms of trafficking. Sexual exploitation, forced labour, servitude, kidnapping for forced prostitution, rape - the list of atrocities committed in armed conflict is, unfortunately, long. Deriving profits from <VIC>human beings</VIC> and considering <VIC>them</VIC> as merchandise, <VIL>the traffickers, as well as consumers and users and the clients of sexual exploitation</VIL>, clearly and brutally violate their <VIL>victims'</VIL> human rights and further stoke the causes of conflict.  <HER>The international community</HER> has invested heavily in this problem since the beginning of this century, but further efforts are needed in order to address the scourge of human trafficking. I therefore welcome the initiative of <HER>the United Kingdom</HER> during its presidency of the Security Council and thank <HER>that country</HER> for giving us this opportunity to have an exchange on this very important subject.  For <HER>France</HER>, the issue oftrafficking in <VIC>human beings</VIC> and slavery, especially that of <VIC>women and children</VIC>, is of major importance. The statistics are, unfortunately, well known, but we must constantly point them out: 80 per cent of the victims of trafficking are <VIC>women and children</VIC>. The challenges are also well known: the identification of <VIC>victims</VIC> is still in its infancy, and organized mechanisms for fighting this scourge vary greatly between countries. Despite progress since the entry into force of the Palermo Protocol to Prevent, Suppress and Punish Trafficking in Persons, Especially Women and Children, still too few prosecutions have been initiated in cases involving the crime of human trafficking. <VIC>The victims</VIC> themselves do not always assert their rights and very often are insufficiently protected. In that context, international cooperation must be stepped up so as to increase the geographic coverage of the legislation providing effective protection against <VIL>networks</VIL> and to improve international cooperation aimed at dismantling <VIL>those networks</VIL>. Prevention, protection and the fight against impunity are the three priorities of French diplomacy in the fight against trafficking in <VIC>human beings</VIC>.  Since human trafficking is now an integral part of the strategy of <VIL>certain terrorist groups</VIL> and it fuels transnational organized crime, the Security Council has a special responsibility in combating this scourge. The adoption of resolution 2331 (2016), last December, at the initiative of <HER>Spain</HER>, was a major step forward towards better addressing the link between trafficking in <VIC>human beings</VIC>, sexual violence and terrorism. France very much looks forward to the report to be prepared by the Secretary-General by the end of the year. <HER>We</HER> have in place a robust international legal framework and appropriate tools, in particular the United Nations Convention against Transnational Organized Crime and its Protocols and <HER>the United Nations Office on Drugs and Crime</HER>, which is doing sterling work in this field.  On International Women's Rights Day, <HER>the President of the French Republic</HER> also announced that <HER>France</HER> would propose an additional protocol to the Convention for the Elimination of All Forms of Discrimination against Women. That protocol would address violence against <VIC>women</VIC> in order to complement the existing international framework. But we must ensure that the obligations arising from that legal framework are effectively implemented. Our words must now be translated into action.  Rest assured, Mr. President, that <HER>France</HER> will continue to play its full part in those efforts.”

Annotated Example 2: 
Mr. Juwayeyi (Malawi): <VIC>Those of us who are not  in the Security Council</VIC> do not get an opportunity to  congratulate a delegation for assuming the presidency  of the Council, so it gives me particular pleasure this  morning to congratulate <HER>you, Mr. President</HER>, and I am  most grateful that during your presidency <HER>you</HER> have  taken the initiative to hold this open session on women  and peace and security.    <HER>My Government</HER> attaches great importance to the  protection and security of <VIC>women and girls</VIC>, both in  situations of armed conflict and in peace. Wars and  armed conflict bring untold suffering and misery to  <VIC>communities and nations</VIC>, for they entail devastating  and horrific levels of violence and brutality, employing  any possible means. Today's wars and conflicts make  little distinction between <VIC>soldiers and civilians</VIC> and  between <VIC>adults and children</VIC>. Currently, most of the  wars and conflicts take place in developing countries,  where most of the population lives in rural areas.  Often, these conflicts are within countries, rather than  across borders. <VIC>Women and children</VIC> constitute a  disproportionate number of the affected populations  and, therefore, suffer the brunt of violence and  brutality.    Armed conflict affects <VIC>women and girls</VIC>  differently from <VIC>men and boys</VIC>. During armed conflict,  not only are <VIC>women and girls</VIC> killed, maimed, abducted,  separated from their loved ones, subjected to  starvation, malnutrition and forced displacement, but  <VIC>they</VIC> are also continually threatened with rape,  domestic violence, sexual exploitation and slavery,  trafficking, sexual humiliation and mutilation. Rape  and sexual violence perpetrated by <VIL>the armed forces,  whether governmental or other actors, including in  some instances peacekeeping personnel</VIL>, increases the  potential for spreading HIV/AIDS and other sexually  transmitted diseases. No wonder most of the  HIV/AIDS victims in the developing countries are  <VIC>women and girls</VIC>. HIV/AIDS leaves <VIC>millions of  children</VIC> orphaned and, in most cases, the responsibility  to care for <VIC>them</VIC> rests largely on the shoulders of <HER_VIC>older  people</HER_VIC>.    All of these harmful and widespread threats to  <VIC>women and girls</VIC> have long-term consequences for  durable peace, security and development. The sad thing  is that in most instances <VIC>the women</VIC> do not know why  the wars and armed conflicts erupt, owing to the fact  that <VIC>they</VIC> are either under-represented or not  represented at all at the decision-making levels.    My Government applauds and thanks <HER>the  Secretary-General, the United Nations bodies and  agencies, non-governmental organizations,  international agencies and donor countries</HER> for the  efforts they have made to protect and secure peace and  security for <VIC>women and girls</VIC>. Various international  legal instruments, particularly the Convention on the  Elimination of All Forms of Discrimination against  Women, the Convention on the Rights of the Child and  the corresponding Optional Protocols, have been  adopted by <HER>the General Assembly</HER>. <HER>International  Criminal Tribunals</HER> have been established for the  former Yugoslavia and for Rwanda, and these have  made great strides to help end impunity for crimes  against <VIC>women and girls</VIC>. Forms of sexual violence are  now included as a war crime in the Rome Statute of the  International Criminal Court. Beyond its emergency  relief responsibilities, <HER>the United Nations</HER> has  increasingly become involved in efforts aimed at  peacekeeping, peace-making and peace-building. It is  gratifying to note that <HER_VIL>the Security Council</HER_VIL>, even  though it has taken five decades to do so, has now  recognized the importance of <HER_VIC>women's</HER_VIC> role and of their  increased participation in the prevention and resolution  of conflicts and in peace-building.    However, there is still a lot more that needs to be  done. Appropriate solutions cannot be achieved if  <VIC>women</VIC> are left out of the decision-making machinery.  <VIL>You</VIL> are aware, <VIL>Mr. President</VIL>, that <VIC>women</VIC> continue to  be under-represented in all peacekeeping, peace-  making and peace-building efforts, including in the  Department of Peacekeeping Operations in the United  Nations. This should not be allowed to continue.    In the long term however, the only way to truly  ensure the protection and security of <VIC>women and girls</VIC>  is to prevent wars and armed conflicts from taking  place. Major root causes of most of the recent wars and  armed conflicts have included poverty and lack of  respect for human rights. These ills must be addressed  first. My delegation recalls that, at the end of the  Millennium Summit, world leaders pledged to  eradicate poverty and make the right to development a  reality for everyone. This means promoting equality  between men and women in decision-making. This  further means the involvement and full participation of  <HER_VIC>women</HER_VIC> in all issues, including peacekeeping,  peacemaking and peace-building, as well as at the  negotiating table, from the grassroots level to the  decision-making levels.    <HER>My delegation</HER> requests the Secretary-General and  the Security Council to urge Member States to ensure  that training in human rights and peacekeeping,  peacemaking and peace-building includes everyone -  civilians, soldiers, the police, civil society, the women  themselves and peacekeeping personnel. <VIL>Those who  commit crimes against <VIC>women</VIC>, including the  peacekeeping personnel</VIL>, should be brought to book.    Let <HER>us</HER> heed the <VIC>women's</VIC> cry for an equal  opportunity to voice their ideas in official peace  negotiations. And let <HER>us</HER> act now.

INSTRUCTION: Rewrite the entire speech with the in-line tags included as specified. When annotating, do not rely on any external world knowledge about the entities described in the text. Focus on the way that the speaker portrays the entities in the text. In your output, do not provide any additional explanation or formatting. Make sure to include all spaces from the original text. Multiple spaces are not equivalent to line breaks. Keep the exact same number of spaces between words. Do not replace spaces with new lines. Do not change capitalization, punctuation, or spelling even if you find mistakes. Only include the annotated text—nothing else. The output should strictly be the speech with the appropriate tags for the identified characters.

Annotate this speech: 
{}"""
            

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
    output_filename = f"./predictions/{model_name}_{prompt_method}_predictions.json"

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
            "zero", # "zero" or "few"
            "./data/train_dev_test_split/test.json")
    

