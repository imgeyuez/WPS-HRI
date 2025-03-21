# pip install ollama

import ollama 


speech = "Women who were raped by armed forces fight for their rights. And I am here so support them!"

prompt= f"""
It is going to be your task to identify the roles of Victim, Hero and Villain in a given speech. 

Begin of Annotation Guidelines:
1. Character Level Rules:
a. annotation has to consist of the whole NP in order to include as much information as possible when identifying characters and their roles; some examples include numbers. Example: “60 million Africans,” “60 percent”
b. Descriptive clauses: “parties that perpetuate acts of violence against women and children”
c. Self-oriented possessive modifiers: “my delegation”

2. Role-Definitions:
a. Hero: Heroes are people who, by helping victims (and defeating the villains), can become heroes. They are defined as people who increase agreement within groups and boost commitment to a cause. They tend to be well-intentioned people, who recognize injustice, try to resolve and fight it, as well as protect others. However, this does not mean that heroes are completely independent. Jaspers et al. (2018) state that even a hero might be in need of help from an even more experienced hero. Furthermore, they are often put in the context of success. 
Task specific additions:
Someone who has the potential to save others, given their abilities, knowledge, positionality/perspective, but who is not able to do so because of structural discrimination
Someone who makes a call to action
Someone who is recognising the unjust treatment and violation of victims or is calling it out.


b. Victim: Victims tend to be portrayed as weak, good, innocent people who are in need of protection. Due to these characteristics, they often motivate and encourage action towards a specific cause and can help make aware of injustices which are worth combatting. Jaspers et al. (2018) state that victim’s sufferings are often elaborated in detail to arouse more moral emotions and indignation. “Popular” victims, as they get the most sympathetic reactions in the modern world due to their cultural innocence, are children. 
Task-specific additions:
Someone who is excluded from decision-making processes/someone who is not given the recognition/power that they deserve
Someone who suffers acts of sexual violence/physical harm/displacement, etc. 
Someone who is not given the same equal rights as other parties

c. Villain: Villains are people whose moral reputation turns or has turned negative. They are considered to be people who spread anxiety and fear, cause people to lose their daily routines, and make them sacrifice their lives, for example, within wars. Perpetrators often share the same characteristics as heroes, such as being strong, brave, and intelligent. However, their description tends to be more like that of beast-like predators: powerful, threatening, and delinquent.
Task specific additions: 
Someone who is responsible for causing anxieties, damage, and crimes. 
Someone who is the cause of people losing their daily routines. 
Someone who stands in the way of equal rights and justice for victims

3. Entities to tag:
a. People b. Organisations  c. Countries d. Groups  f. No laws etc. except for mentioned UNSC Resolutions

4. What not to annotate:
a. Entities who are hoping for something good to happen. We identify hope as a passive action, showing too little initiative to actually change anything in order to be considered as a hero. 
b. Entities welcoming something
c. Expressions of thanking/commending something are to be treated similar to the ones of hoping.

5. Example:
“We commend the work that has been done by the United Nations Children's Fund in reintegration projects that has led to the release of girls from the armed forces in various countries.” 

Only generate an output in the format as shown in the Example. 

Annotate the following speech:
"{speech}"
"""



stream = ollama.chat(
    model='deepseek-r1', # name of llama to use
    messages=[{'role': 'user', 'content': prompt}],
    stream=True # if true, output is printed word by word 
)


# loop over all chunks of output stream and give them out together
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
