from nltk.sem.drt import *
import clfutils
import re # helps with adding tense

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
        elif line[1]=='ARG2_VERB' or line[1]=='ARG_OBJ' or line[1]=='ARG2' or line[1]=='ARG1_AUX':
            line[1] = 'Theme'
        elif line[1]=='ARG2_AUX':
            line[1] = 'Co-Theme'
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
        elif line[1]=='NUM_RELATION':
            line[1] = 'Quantity'
        elif line[1]=='NUM_PROPERTY':
            line[1] = 'quantity.n.01'
    return [' '.join(x) for x in clflinessplit]

# Guess what tense any verbs/auxiliaries are in and adds the times for those.
def guess_tenses(clflines):
    # Time(x,t),TIMEREL(t,"now"),FINDWORDNET_ROOT_time(t)
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
            elif "VerbForm;Fin" in line[1]:
                timevar = f"t{nextnumber}"
                nextnumber = nextnumber + 1
                extralines.append(f"{line[0]} Time {verbvar} {timevar}")
                extralines.append(f'{line[0]} time "n.08" {timevar}')
                extralines.append(f'{line[0]} TPR {timevar} "now"')
    return clflines + extralines

# Guesses missing information about proper nouns. (Assigning all to person.n.02 as UD-Boxer does.)
def guess_propn_type(clflines):
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


def postprocess_clf(clflines):
    return guess_wordnet_synsets(
            guess_missing_relations(
            guess_tenses(
            guess_propn_type(
            unclean_lemmas(clflines)))))

# DRS version of the previous; currently a bit of a cheater.
def postprocess_drs(drs):
    return clfutils.clf_to_drs(postprocess_clf(clfutils.drs_to_clf(drs)))
