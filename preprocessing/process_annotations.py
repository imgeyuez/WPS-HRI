################# MODULES ################# 
import os
import copy 
import json
import random


################# FUNCTIONS #################
def train_dev_test_split():
    """
    This function separates a list of filenames
    and decides which ones are to be used as 
    train, dev and test data.
    """

    filenames = """UNSC_2000_SPV.4208Resumption1_spch030.txt
        UNSC_2000_SPV.4208Resumption2_spch004.txt
        UNSC_2000_SPV.4208_spch004.txt
        UNSC_2000_SPV.4208_spch010.txt
        UNSC_2002_SPV.4589Resumption1_spch011.txt
        UNSC_2002_SPV.4589Resumption1_spch015.txt
        UNSC_2003_SPV.4852Resumption1_spch004.txt
        UNSC_2003_SPV.4852Resumption1_spch018.txt
        UNSC_2003_SPV.4852Resumption1_spch035.txt
        UNSC_2004_SPV.5066Resumption1_spch006.txt
        UNSC_2004_SPV.5066Resumption1_spch026.txt
        UNSC_2005_SPV.5294Resumption1_spch036.txt
        UNSC_2005_SPV.5294Resumption1_spch046.txt
        UNSC_2005_SPV.5294_spch027.txt
        UNSC_2006_SPV.5556Resumption1_spch002.txt
        UNSC_2006_SPV.5556_spch002.txt
        UNSC_2007_SPV.5766Resumption1_spch020.txt
        UNSC_2007_SPV.5766Resumption1_spch034.txt
        UNSC_2008_SPV.6005Resumption1_spch020.txt
        UNSC_2008_SPV.6005Resumption1_spch058.txt
        UNSC_2008_SPV.6005_spch029.txt
        UNSC_2009_SPV.6195_spch008.txt
        UNSC_2009_SPV.6196Resumption1_spch004.txt
        UNSC_2010_SPV.6302_spch019.txt
        UNSC_2010_SPV.6411Resumption1_spch064.txt
        UNSC_2010_SPV.6411_spch053.txt
        UNSC_2011_SPV.6642Resumption1_spch052.txt
        UNSC_2011_SPV.6642_spch055.txt
        UNSC_2012_SPV.6722Resumption1_spch012.txt
        UNSC_2012_SPV.6722_spch023.txt
        UNSC_2012_SPV.6877_spch051.txt
        UNSC_2013_SPV.6948_spch067.txt
        UNSC_2013_SPV.6984_spch069.txt
        UNSC_2013_SPV.7044_spch013.txt
        UNSC_2013_SPV.7044_spch049.txt
        UNSC_2014_SPV.7160_spch008.txt
        UNSC_2014_SPV.7289_spch006.txt
        UNSC_2014_SPV.7289_spch107.txt
        UNSC_2015_SPV.7374_spch123.txt
        UNSC_2015_SPV.7428_spch043.txt
        UNSC_2015_SPV.7533_spch030.txt
        UNSC_2015_SPV.7585_spch010.txt
        UNSC_2016_SPV.7643_spch010.txt
        UNSC_2016_SPV.7704_spch087.txt
        UNSC_2016_SPV.7847_spch004.txt
        UNSC_2017_SPV.7898_spch010.txt
        UNSC_2017_SPV.7938_spch066.txt
        UNSC_2017_SPV.8079_spch078.txt
        UNSC_2017_SPV.8079_spch118.txt
        UNSC_2018_SPV.8234_spch045.txt
        UNSC_2018_SPV.8382_spch016.txt
        UNSC_2019_SPV.8514_spch023.txt
        UNSC_2019_SPV.8649_spch008.txt
        UNSC_2019_SPV.8657_spch018.txt"""

    previous_annotated_speeches = ['UNSC_2000_SPV.4208Resumption1_spch030.txt',
                               'UNSC_2007_SPV.5766Resumption1_spch020.txt',
                               'UNSC_2008_SPV.6005Resumption1_spch020.txt',
                               'UNSC_2000_SPV.4208Resumption2_spch004.txt']
    
    cleaned_files = [name.strip() for name in filenames if name.strip() not in previous_annotated_speeches]
    
    random.shuffle(cleaned_files)

    train = cleaned_files[0:40] + previous_annotated_speeches
    dev = cleaned_files[40:45]
    test = cleaned_files[45:]

    return train, dev, test

def read_tsv_files(backup_path):
    
    files = list()

    if os.path.exists(backup_path) and os.path.isdir(backup_path):
        if os.path.isdir(backup_path):
            # Iterate through all folders in the backup
            for folder in os.listdir(backup_path):
                folder_path = os.path.join(backup_path, folder)
                for file in os.listdir(folder_path):
                    if file.endswith('.tsv'):

                        # create a dict for the file
                        speech = {
                            'filename': folder,
                            'raw_file_input': list(),
                            'sentences': list()
                        }
                        with open(os.path.join(folder_path, file), 'r', encoding="UTF-8") as tsvfile:
                            data = tsvfile.readlines()
                        
                            for line in data:
                                speech['raw_file_input'].append(line.split("\t"))
                                
                        files.append(speech)
                            
    return files      

def label_token(prev_label, current_label, sentence_goldlabels):
    """
    Updates sentence labels based on the current and previous tags.
    Differentiates between B- and I- labels.
    """
    if prev_label == current_label:
        sentence_goldlabels.append(f"I-{current_label}")  # Inside the same label
    elif prev_label is not None and current_label in prev_label:
        sentence_goldlabels.append(f"I-{current_label}")
    else:
        sentence_goldlabels.append(f"B-{current_label}")  # Beginning a new label

def clean_label(label):
    if "|" in label:
        labels = label.split("|")
        current_labels = list()
        for l in labels:
            current_labels.append(l.split('[')[0])
        label = '_'.join(current_labels)

        if label == "VILLAIN_VICTIM":
            label = "VICTIM_VILLAIN"
        elif label == "VILLAIN_HERO":
            label = "HERO_VILLAIN"
        elif label == "VICTIM_HERO":
            label = "HERO_VICTIM"
        
    else:
        label = label.split('[')[0]

    return label

def process_file(file_copy, file_original):
    """
    Processes a single file to extract sentences and their labels.
    """
    sentence_tokens, sentence_goldlabels, sentence_offset = [], [], []

    for index, line in enumerate(file_original["raw_file_input"]):

        # get  previous label
        prev_label = None if index == 0 else file_original["raw_file_input"][index - 1][3] if len(file_original["raw_file_input"][index - 1]) > 4 else None
        
        if prev_label:
            prev_label = clean_label(prev_label)

        if len(line) == 1:  # New sentence
            if sentence_tokens:
                if len(sentence_tokens) != len(sentence_goldlabels):
                    raise ValueError("MISMATCH")
                file_copy["sentences"].append({
                    "tokens": sentence_tokens,
                    "goldlabels": sentence_goldlabels,
                    "offset": sentence_offset,
                })
            sentence_tokens, sentence_goldlabels, sentence_offset = [], [], []

        elif len(line) == 5:  # Process tokens
            token = line[2]
            token_offset = tuple(line[1].split("-"))
            sentence_tokens.append(token)
            sentence_offset.append(token_offset)

            if line[3] == "_":
                sentence_goldlabels.append("O")
            else:
                current_label = clean_label(line[3])
                
                label_token(prev_label, current_label, sentence_goldlabels)

def json_output(files_copy, data_type):

    output_path = f'./{data_type}.json'
    
    for file in files_copy:
        file.pop("raw_file_input", None)  # Remove the field before writing to JSON
    
    with open(output_path, "w") as pred:    
        json.dump(files_copy, pred, indent=4)
    
def process_files(files, train, dev, test):
    """
    Processes all files and adds labeled sentences.
    """
    files_copy = copy.deepcopy(files)  # Create a deep copy to modify
    
    for file_original, file_copy in zip(files, files_copy):
        process_file(file_copy, file_original)

    train_data, dev_data, test_data = [], [], []

    for file in files_copy:
        filename = file["filename"]
        if filename in train:
            train_data.append(file)
        elif filename in dev:
            dev_data.append(file)
        else:
            test_data.append(file)


    json_output(train_data, "train")
    json_output(dev_data, "dev")
    json_output(test_data, "test")


################# CODE #################

random.seed(9)

train, dev, test = train_dev_test_split()

"""
# Train data
['UNSC_2016_SPV.7704_spch087.txt', 'UNSC_2005_SPV.5294_spch027.txt', 'UNSC_2016_SPV.7847_spch004.txt', 'UNSC_2004_SPV.5066Resumption1_spch006.txt', 'UNSC_2002_SPV.4589Resumption1_spch011.txt', 'UNSC_2014_SPV.7289_spch006.txt', 'UNSC_2011_SPV.6642_spch055.txt', 'UNSC_2014_SPV.7289_spch107.txt', 'UNSC_2002_SPV.4589Resumption1_spch015.txt', 'UNSC_2009_SPV.6195_spch008.txt', 'UNSC_2000_SPV.4208_spch004.txt', 'UNSC_2015_SPV.7374_spch123.txt', 'UNSC_2006_SPV.5556_spch002.txt', 'UNSC_2008_SPV.6005Resumption1_spch058.txt', 'UNSC_2011_SPV.6642Resumption1_spch052.txt', 'UNSC_2005_SPV.5294Resumption1_spch046.txt', 'UNSC_2017_SPV.7938_spch066.txt', 'UNSC_2009_SPV.6196Resumption1_spch004.txt', 'UNSC_2003_SPV.4852Resumption1_spch004.txt', 'UNSC_2015_SPV.7428_spch043.txt', 'UNSC_2013_SPV.7044_spch013.txt', 'UNSC_2018_SPV.8382_spch016.txt', 'UNSC_2010_SPV.6302_spch019.txt', 'UNSC_2013_SPV.7044_spch049.txt', 'UNSC_2017_SPV.8079_spch078.txt', 'UNSC_2006_SPV.5556Resumption1_spch002.txt', 'UNSC_2012_SPV.6877_spch051.txt', 'UNSC_2005_SPV.5294Resumption1_spch036.txt', 'UNSC_2012_SPV.6722Resumption1_spch012.txt', 'UNSC_2019_SPV.8514_spch023.txt', 'UNSC_2019_SPV.8649_spch008.txt', 'UNSC_2004_SPV.5066Resumption1_spch026.txt', 'UNSC_2012_SPV.6722_spch023.txt', 'UNSC_2019_SPV.8657_spch018.txt', 'UNSC_2008_SPV.6005_spch029.txt', 'UNSC_2017_SPV.7898_spch010.txt', 'UNSC_2017_SPV.8079_spch118.txt', 'UNSC_2003_SPV.4852Resumption1_spch018.txt', 'UNSC_2010_SPV.6411_spch053.txt', 'UNSC_2018_SPV.8234_spch045.txt', 'UNSC_2000_SPV.4208Resumption1_spch030.txt', 'UNSC_2007_SPV.5766Resumption1_spch020.txt', 'UNSC_2008_SPV.6005Resumption1_spch020.txt', 'UNSC_2000_SPV.4208Resumption2_spch004.txt']

# Dev data
['UNSC_2016_SPV.7643_spch010.txt', 'UNSC_2010_SPV.6411Resumption1_spch064.txt', 'UNSC_2015_SPV.7585_spch010.txt', 'UNSC_2003_SPV.4852Resumption1_spch035.txt', 'UNSC_2015_SPV.7533_spch030.txt']

# Test data
['UNSC_2014_SPV.7160_spch008.txt', 'UNSC_2013_SPV.6984_spch069.txt', 'UNSC_2000_SPV.4208_spch010.txt', 'UNSC_2007_SPV.5766Resumption1_spch034.txt', 'UNSC_2013_SPV.6948_spch067.txt']
"""

# changeable variables
backup_path = './inception_curation_backup/curation'

files = read_tsv_files(backup_path=backup_path)

num_of_cols = set()
for file in files:
    for line in file["raw_file_input"]:
        num_of_cols.add(len(line))

process_files(files, train, dev, test)