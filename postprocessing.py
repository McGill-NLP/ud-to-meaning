from nltk.sem.drt import *
import clfutils
import re # helps with adding tense

tokensubs = []
with open("tokensubs.csv") as f:
    tokensubs = [x.split() for x in f.read().split('\n') if '\t' in x]

def unclean_lemmas(clflines):
    '''
    Undoes the clean_lemmas step from preprocessing.
    Takes a clf representing a DRS, and adds back in the characters
    that we had to remove because they wouldn't play well with derivation.
    
            Parameters:
                    clflines: A list of strings representing the lines of the CLF representation of DRS.

            Returns:
                    newlines: A list of strings representing the re-substituted CLF of the DRS.
    '''
    newlines = []
    newline = ""
    for line in clflines:
        newline = line
        if len(newline.split())>1 and newline.split()[1].endswith("_LETTER"):
            newline = newline.split()[0] + ' ' + newline.split()[1][:-7] + ' ' + ' '.join(newline.split()[2:])
        for sub in tokensubs:
            newline = newline.replace(sub[1],sub[0])
        newlines.append(newline)
    return newlines

def guess_wordnet_synsets(clflines,synsetdict={}):
    '''
    Guesses the wordnet synsets corresponding to each predicate in the DRS.
    Just guesses "01" for everything, unless you pass a synsetdict with better guesses.
    
            Parameters:
                    clflines: A list of strings representing the lines of the CLF representation of DRS.
                    synsetdict: Optional. A list of known "best guess" word-to-synset mappings, if you have one. These guesses will be used for these words.

            Returns:
                    newlines: A list of strings representing the re-substituted CLF of the DRS with synset info added.
    '''
    clflinessplit = [[x.split()[0],x.split()[1],' '.join(x.split()[2:])] for x in clflines if len(x.split())>1]
    for line in clflinessplit:
        if line[1].startswith("FINDWORDNET"):
            lemma = '_'.join(line[1].split('_')[3:])
            POS = line[1].split('_')[1]
            feats = line[1].split('_')[2]
            POSshort = ""
            if POS=='VERB' or POS=='AUX':
                POSshort='v'
            if POS=='NOUN':
                POSshort='n'
            if POS=='ADV':
                POSshort='r'
            if POS=='ADJ':
                POSshort='a'
            if lemma+"."+POSshort in synsetdict:
                synset = synsetdict[lemma+"."+POSshort]
                line[1] = f'{".".join(synset.split(".")[:-2])} "{".".join(synset.split(".")[-2:])}"'
            elif lemma in synsetdict:
                synset = synsetdict[lemma]
                line[1] = f'{".".join(synset.split(".")[:-2])} "{".".join(synset.split(".")[-2:])}"'
            else:
                line[1]= f'{lemma} "{POSshort}.01"'
    return [' '.join(x) for x in clflinessplit]

def guess_missing_relations(clflines):
    '''
    Guesses relation labels for verbs, adjectives, and possessors, tense based on most frequent class
    possessors = User
    subject/Arg1 = Agent
    object/Arg2 = Theme
    Arg3 = Recipient
    Adjective property = Attribute
    Adjective Arg2 = Stimulus
    Num = Number
    
            Parameters:
                    clflines: A list of strings representing the lines of the CLF representation of DRS.

            Returns:
                    newlines: A list of strings representing the re-substituted CLF of the DRS with argument roles added.
    '''
    clflinessplit = [[x.split()[0],x.split()[1],' '.join(x.split()[2:])] for x in clflines if len(x.split())>1]
    for line in clflinessplit:
        if line[1]=='ARG_POSS':
            line[1] = 'User'
        elif line[1]=='ARG1_VERB' or line[1]=='ARG_NSUBJ' or line[1]=='ARG1':
            line[1] = 'Agent'
        elif line[1]=='ARG2_VERB' or line[1]=='ARG_OBJ' or line[1]=='ARG2' or line[1]=='ARG1_AUX':
            line[1] = 'Theme'
        elif line[1]=='ARG2_AUX':
            line[1] = 'Co-Theme'
        elif line[1]=='ARG3' or line[1]=='ARG3_VERB':
            line[1] = 'Recipient'
        elif line[1]=='ADJ_RELATION':
            line[1] = 'Attribute'
        elif line[1].startswith('ADP_RELATION'):
            line[1] = 'Location'
        elif line[1]=='ADV_RELATION':
            line[1] = 'Manner'
        elif line[1]=='ARG2_ADJ':
            line[1] = 'Stimulus'
        elif line[1]=='TIMEREL':
            line[1] = 'TPR'
        elif line[1]=='ARG_NPMOD':
            line[1] = 'Quantity'
        elif line[1]=='ARG_TMOD':
            line[1] = 'Quantity'
        elif line[1]=='NUM_RELATION':
            line[1] = 'Quantity'
        elif line[1]=='NUM_PROPERTY':
            line[1] = 'quantity.n.01'
    return [' '.join(x) for x in clflinessplit]

def guess_tenses(clflines):
    '''
    Guess what tense any verbs/auxiliaries are in and adds the times for those to a computed DRS.
    
            Parameters:
                    clflines: A list of strings representing the lines of the CLF representation of DRS.

            Returns:
                    newlines: A list of strings representing the re-substituted CLF of the DRS with temporal info added.
    '''
    clflinessplit = [[x.split()[0],x.split()[1],' '.join(x.split()[2:])] for x in clflines if len(x.split())>1]
    usednumbers = [int(x) for x in re.findall(r'\d+',' '.join(clflines))]
    nextnumber = max(usednumbers)+1
    extralines = []
    for line in clflinessplit:
        if line[1].startswith('FINDWORDNET_VERB') or line[1].startswith('FINDWORDNET_AUX'):
            verbvar = line[2].split()[0]
            if "Tense;Past" in line[1]:
                timevar = f"t{nextnumber}"
                nextnumber = nextnumber + 1
                extralines.append(f"{line[0]} Time {verbvar} {timevar}")
                extralines.append(f'{line[0]} time "n.08" {timevar}')
                extralines.append(f'{line[0]} TPR {timevar} "now"')
            elif "Tense;Pres" in line[1]:
                timevar = f"t{nextnumber}"
                nextnumber = nextnumber + 1
                extralines.append(f"{line[0]} Time {verbvar} {timevar}")
                extralines.append(f'{line[0]} time "n.08" {timevar}')
                extralines.append(f'{line[0]} EQU {timevar} "now"')
            elif "Tense;Fut" in line[1]:
                timevar = f"t{nextnumber}"
                nextnumber = nextnumber + 1
                extralines.append(f"{line[0]} Time {verbvar} {timevar}")
                extralines.append(f'{line[0]} time "n.08" {timevar}')
                extralines.append(f'{line[0]} TSU {timevar} "now"')
            elif "VerbF_OR_m;Fin" in line[1]:
                timevar = f"t{nextnumber}"
                nextnumber = nextnumber + 1
                extralines.append(f"{line[0]} Time {verbvar} {timevar}")
                extralines.append(f'{line[0]} time "n.08" {timevar}')
                extralines.append(f'{line[0]} TPR {timevar} "now"')
    return clflines + extralines

def guess_propn_type(clflines):
    '''
    Guess what sort of thing (person, place?) proper nouns refer to.
    
            Parameters:
                    clflines: A list of strings representing the lines of the CLF representation of DRS.

            Returns:
                    newlines: A list of strings representing the re-substituted CLF of the DRS with proper noun added.
    '''
    clflinessplit = [[x.split()[0],x.split()[1],' '.join(x.split()[2:])] for x in clflines if len(x.split())>1]
    extralines = []
    for line in clflinessplit:
        if line[1].startswith('PROPN_PROPERTY'):
            if "Gender;Fem" in line[1]:
                line[1] = 'female "n.02"'
            elif "Gender;Masc" in line[1]:
                line[1] = 'male "n.02"'
            elif "Gender;Neut" in line[1]:
                line[1] = 'entity "n.01"'
            else:
                line[1] = 'person "n.01"'
        if line[1].startswith('PRONOUN_PROPERTY'):
            if "Person;1" in line[1]:
                if "Number;Plur" in line[1]:
                    # we
                    line[1] = 'person "n.01"'
                    # add a this-guy-Sub-speaker line
                    extralines.append(' '.join((line[0],'Sub',line[2].split()[0],'"speaker"',' '.join(line[2].split()[1:]))))
                else:
                    # I
                    line[1] = 'person "n.01"'
                    # add a this-guy-EQU-speaker line
                    extralines.append(' '.join((line[0],'EQU',line[2].split()[0],'"speaker"',' '.join(line[2].split()[1:]))))
            elif "Person;2" in line[1]:
                # you
                line[1] = 'person "n.01"'
                # add a this-guy-EQU-hearer line
                extralines.append(' '.join((line[0],'EQU',line[2].split()[0],'"hearer"',' '.join(line[2].split()[1:]))))
            elif "Gender;Fem" in line[1]:
                line[1] = 'female "n.02"'
            elif "Gender;Masc" in line[1]:
                line[1] = 'male "n.02"'
            elif "Gender;Neut" in line[1]:
                line[1] = 'entity "n.01"'
            else:
                line[1] = 'person "n.01"'
    return [' '.join(x) for x in clflinessplit] + extralines


def postprocess_clf(clflines,synsetdict={}):
    '''
    Performs all of the postprocessing steps from postprocess_clf, but
    takes a DRS (from NLTK DRT package) as input and returns one.
    
            Parameters:
                    clflines: A list of strings representing the lines of the CLF representation of DRS.

            Returns:
                    newlines: A list of strings representing the fully post-processed DRS.
    '''
    return guess_wordnet_synsets(
            guess_missing_relations(
            guess_tenses(
            guess_propn_type(
            unclean_lemmas(clflines)))),synsetdict)

# DRS version of the previous; currently a bit of a cheater.
def postprocess_drs(drs,synsetdict={}):
    '''
    Performs all of the postprocessing steps from postprocess_clf, but
    takes a DRS (from NLTK DRT package) as input and returns one.
    
            Parameters:
                    drs: A list of strings representing the lines of the CLF representation of DRS.
                    synsetdict: The dictionary of most frequent synsets to use to guess words' synsets, if any.

            Returns:
                    drs: A new DRS with all postprocessing steps done.
    '''
    return clfutils.clf_to_drs(postprocess_clf(clfutils.drs_to_clf(drs),synsetdict))
