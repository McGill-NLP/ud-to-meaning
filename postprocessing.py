from nltk.sem.drt import *

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

def postprocess_clf(clflines):
    return unclean_lemmas(clflines)
