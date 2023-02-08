from nltk.sem.drt import *
import clfutils

tokensubs = []
with open("tokensubs.csv") as f:
    tokensubs = [x.split() for x in f.read().split('\n') if '\t' in x]

# TODO add DRS editions of all postprocessing functions.

# Undoes the word substitutions that happend in Preprocessing (CLF edition)
def unclean_lemmas(clflines):
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

# Guesses wordnet synsets for tenses coming from lemmas
# Currently just guesses "01" for all of them, as Poelman mentions would be reasonable.
def guess_wordnet_synsets(clflines):
    clflinessplit = [[x.split()[0],x.split()[1],' '.join(x.split()[2:])] for x in clflines if len(x.split())>1]
    for line in clflinessplit:
        if line[1]=="FINDWORDNET_ROOT_time":
            line[1]='time "n.08"'
        elif line[1].startswith("FINDWORDNET"):
            lemma = '_'.join(line[1].split('_')[2:])
            POS = line[1].split('_')[1]
            POSshort = ""
            if POS=='VERB':
                POSshort='v'
            if POS=='NOUN':
                POSshort='n'
            if POS=='ADV':
                POSshort='r'
            if POS=='ADJ':
                POSshort='a'
            line[1]= f'{lemma} "{POSshort}.01"'
    return [' '.join(x) for x in clflinessplit]

# Guesses relation labels for verbs, adjectives, and possessors, tense based on most frequent class
# possessors = User
# subject/Arg1 = Agent
# object/Arg2 = Theme
# Arg3 = Recipient
# Adjective property = Attribute
# Adjective Arg2 = Stimulus
# Num = Number
def guess_missing_relations(clflines):
    clflinessplit = [[x.split()[0],x.split()[1],' '.join(x.split()[2:])] for x in clflines if len(x.split())>1]
    for line in clflinessplit:
        if line[1]=='ARG_POSS':
            line[1] = 'User'
        elif line[1]=='ARG1_VERB' or line[1]=='ARG_NSUBJ' or line[1]=='ARG1':
            line[1] = 'Agent'
        elif line[1]=='ARG2_VERB' or line[1]=='ARG_OBJ' or line[1]=='ARG2':
            line[1] = 'Theme'
        elif line[1]=='ARG3':
            line[1] = 'Recipient'
        elif line[1]=='ADJ_RELATION':
            line[1] = 'Attribute'
        elif line[1]=='ADV_RELATION':
            line[1] = 'Manner'
        elif line[1]=='ARG2_ADJ':
            line[1] = 'Stimulus'
        elif line[1]=='NUM_RELATION':
            line[1] = 'Number'
        elif line[1]=='TIMEREL':
            line[1] = 'TPR'
        elif line[1]=='ARG_NPMOD':
            line[1] = 'Quantity'
        elif line[1]=='ARG_TMOD':
            line[1] = 'Quantity'
    return [' '.join(x) for x in clflinessplit]


# Guesses missing information about proper nouns. (Assigning all to female.n.02 as UD-Boxer does.)
def guess_propn_type(clflines):
    clflinessplit = [[x.split()[0],x.split()[1],' '.join(x.split()[2:])] for x in clflines if len(x.split())>1]
    for line in clflinessplit:
        if line[1]=='PROPN_PROPERTY':
            line[1] = 'female "n.02"'
        if line[1]=='PRONOUN_PROPERTY':
            line[1] = 'female "n.02"'
    return [' '.join(x) for x in clflinessplit]


def postprocess_clf(clflines):
    return guess_wordnet_synsets(
            guess_missing_relations(
                guess_propn_type(
                unclean_lemmas(clflines))))

# DRS version of the previous; currently a bit of a cheater.
def postprocess_drs(drs):
    return clfutils.clf_to_drs(postprocess_clf(clfutils.drs_to_clf(drs)))