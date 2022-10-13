import os # changing working directory
import conllu # reading ConLL-U files and working with the resulting data representation
from nltk.sem.drt import * # working with DRS structures
import stanza # parsing to UD
os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')
# the modules we need should be in the working directory now, and the pmb files.
from semtypes import *
from simpleparsing import *

##### NOTE: I am not including the actual PMB examples in this Git repository,
#####       because they are very big.
#####       You can get them yourself and change the pmbpath below.

pmbpath = "pmb-gold-english"
datapointpath = r"\p00\d0712"

# Get the raw text, tokens, and gold DRT structure.

with open(pmbpath + datapointpath + r"\en.raw") as f:
    datapointraw = f.read()
print(datapointraw)

with open(pmbpath + datapointpath + r"\en.tok.off") as f:
    tokensraw = f.read()
tokens = ['~'.join(x.split(' ')[3:]) for x in tokensraw.split('\n') if len(x) > 0]

with open(pmbpath + datapointpath + r"\en.drs.clf") as f:
    drsclfraw = f.read()

# This takes a raw clause format DRS (the type available in PMB data)
# and outputs a DRS of the class from the DRT module.
# TODO this does not work on DRS that contain other DRS yet.
def clf_to_drs(clfraw):
    clflines = clfraw.split("\n")
    clflinesstripped = [x.split("%")[0].strip() for x in clflines]
    contentlines = [x for x in clflinesstripped if len(x) > 0]
    minuspointers = [x[3:] for x in contentlines if not (x[-2] == 'b' and x[-1].isdigit())]
    refs = []
    conds = []
    for statement in minuspointers:
        if statement.startswith("REF "):
            refs.append(statement[4:])
        else:
            stmtsplit = statement.split(' ')
            argnames = [x for x in stmtsplit[1:] if not(x[1].isalpha() and x[2] == '.' and x[3:-1].isdigit())]
            conds.append(stmtsplit[0] + "(" + ",".join(argnames) + ")")
    return DrtExpression.fromstring("([" + ",".join(refs) + "],[" + ",".join(conds) + "])")

pmb_drs = clf_to_drs(drsclfraw)

# Now we parse the tokens in UD, so we can use the UD-to-meaning parser.
stanzanlp = stanza.Pipeline(lang='en', processors = 'tokenize,mwt,pos,lemma,depparse',tokenize_pretokenized=True)
doc = stanzanlp([tokens])

ud_dict = doc.to_dict()[0]
for x in ud_dict:
    x['form'] = x['text']
ud_parse = conllu.TokenList(ud_dict)

# Then we apply the UD-based meaning program

preprocessed = preprocess(ud_parse)
withdens = conllu.TokenList([add_denotation(token) for token in preprocessed])
simplified = simplifynodetyped(withdens.to_tree())
simpleparsing_drs = simplified.token['word_dens']

print("PMB DRS:\n"+pmb_drs.pretty_format()+"\n\nMy DRS:\n"+simpleparsing_drs[0][1].pretty_format())