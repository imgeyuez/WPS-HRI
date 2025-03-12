import random

random.seed(9)

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
    files = filenames.split("\n")
    cleaned_files = list()
    for file in files:
        if file in previous_annotated_speeches:
            continue
        else:
            cleaned_files.append(file)


    random.shuffle(cleaned_files)
    random.shuffle(cleaned_files)

    train = cleaned_files[0:40]
    train.extend(previous_annotated_speeches)
    dev = cleaned_files[40:45]
    test = cleaned_files[45:]

    return train, dev, test

# train, dev, test = train_dev_test_split()


"""
# Train data
['UNSC_2016_SPV.7704_spch087.txt', 'UNSC_2005_SPV.5294_spch027.txt', 'UNSC_2016_SPV.7847_spch004.txt', 'UNSC_2004_SPV.5066Resumption1_spch006.txt', 'UNSC_2002_SPV.4589Resumption1_spch011.txt', 'UNSC_2014_SPV.7289_spch006.txt', 'UNSC_2011_SPV.6642_spch055.txt', 'UNSC_2014_SPV.7289_spch107.txt', 'UNSC_2002_SPV.4589Resumption1_spch015.txt', 'UNSC_2009_SPV.6195_spch008.txt', 'UNSC_2000_SPV.4208_spch004.txt', 'UNSC_2015_SPV.7374_spch123.txt', 'UNSC_2006_SPV.5556_spch002.txt', 'UNSC_2008_SPV.6005Resumption1_spch058.txt', 'UNSC_2011_SPV.6642Resumption1_spch052.txt', 'UNSC_2005_SPV.5294Resumption1_spch046.txt', 'UNSC_2017_SPV.7938_spch066.txt', 'UNSC_2009_SPV.6196Resumption1_spch004.txt', 'UNSC_2003_SPV.4852Resumption1_spch004.txt', 'UNSC_2015_SPV.7428_spch043.txt', 'UNSC_2013_SPV.7044_spch013.txt', 'UNSC_2018_SPV.8382_spch016.txt', 'UNSC_2010_SPV.6302_spch019.txt', 'UNSC_2013_SPV.7044_spch049.txt', 'UNSC_2017_SPV.8079_spch078.txt', 'UNSC_2006_SPV.5556Resumption1_spch002.txt', 'UNSC_2012_SPV.6877_spch051.txt', 'UNSC_2005_SPV.5294Resumption1_spch036.txt', 'UNSC_2012_SPV.6722Resumption1_spch012.txt', 'UNSC_2019_SPV.8514_spch023.txt', 'UNSC_2019_SPV.8649_spch008.txt', 'UNSC_2004_SPV.5066Resumption1_spch026.txt', 'UNSC_2012_SPV.6722_spch023.txt', 'UNSC_2019_SPV.8657_spch018.txt', 'UNSC_2008_SPV.6005_spch029.txt', 'UNSC_2017_SPV.7898_spch010.txt', 'UNSC_2017_SPV.8079_spch118.txt', 'UNSC_2003_SPV.4852Resumption1_spch018.txt', 'UNSC_2010_SPV.6411_spch053.txt', 'UNSC_2018_SPV.8234_spch045.txt', 'UNSC_2000_SPV.4208Resumption1_spch030.txt', 'UNSC_2007_SPV.5766Resumption1_spch020.txt', 'UNSC_2008_SPV.6005Resumption1_spch020.txt', 'UNSC_2000_SPV.4208Resumption2_spch004.txt']

# Dev data
['UNSC_2016_SPV.7643_spch010.txt', 'UNSC_2010_SPV.6411Resumption1_spch064.txt', 'UNSC_2015_SPV.7585_spch010.txt', 'UNSC_2003_SPV.4852Resumption1_spch035.txt', 'UNSC_2015_SPV.7533_spch030.txt']

# Test data
['UNSC_2014_SPV.7160_spch008.txt', 'UNSC_2013_SPV.6984_spch069.txt', 'UNSC_2000_SPV.4208_spch010.txt', 'UNSC_2007_SPV.5766Resumption1_spch034.txt', 'UNSC_2013_SPV.6948_spch067.txt']
"""