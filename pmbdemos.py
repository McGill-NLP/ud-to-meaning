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

# The NLP pipeline we will use to parse to UD
stanzanlp = stanza.Pipeline(lang='en',
                processors = 'tokenize,mwt,pos,lemma,depparse',
                tokenize_pretokenized=True,
                download_method=stanza.DownloadMethod.REUSE_RESOURCES)

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

# This function just makes demos easier.
def compare_with_pmb(pmbpath, datapointpath, stanzanlp):
    # Get the raw text, tokens, and gold DRT structure.
    with open(pmbpath + datapointpath + r"\en.raw") as f:
        datapointraw = f.read()
    print(datapointraw)

    with open(pmbpath + datapointpath + r"\en.tok.off") as f:
        tokensraw = f.read()
    tokens = ['~'.join(x.split(' ')[3:]) for x in tokensraw.split('\n') if len(x) > 0]

    with open(pmbpath + datapointpath + r"\en.drs.clf") as f:
        drsclfraw = f.read()

    pmb_drs = clf_to_drs(drsclfraw)

    # Now we parse the tokens in UD, so we can use the UD-to-meaning parser.
    doc = stanzanlp([tokens])

    ud_dict = doc.to_dict()[0]
    for x in ud_dict:
        x['form'] = x['text']
    ud_parse = conllu.TokenList(ud_dict)
    print("UD tree:\n")
    ud_parse.to_tree().print_tree()

    # Then we apply the UD-based meaning program
    preprocessed = preprocess(ud_parse)
    withdens = conllu.TokenList([add_denotation(token) for token in preprocessed])
    simplified = simplifynodetyped(withdens.to_tree(),True)
    simpleparsing_drs = simplified.token['word_dens']

    print("\nPMB DRS:\n"+pmb_drs.pretty_format()+"\n\nMy DRS:\n"+simpleparsing_drs[0][1].pretty_format())

    return pmb_drs, simpleparsing_drs

# MARK here is where specific demos begin
# Basic transitive sentence: "Kraft sold Celestial Seasonings"
demo = compare_with_pmb("pmb-gold-english", r"\p00\d0712", stanzanlp)

# Event subject: "Packing sucks"
demo = compare_with_pmb("pmb-gold-english", r"\p00\d1666", stanzanlp)

# Compound noun: "Kohl announced economy measures"
demo = compare_with_pmb("pmb-gold-english", r"\p00\d1222", stanzanlp)

# Intransitive sentence with adverb: "Mary just left"
# demo = compare_with_pmb("pmb-gold-english", r"\p00\d1589", stanzanlp)
# (has a bug in PMB portion)

# Raising: "Tom seems conceited"
demo = compare_with_pmb("pmb-gold-english", r"\p00\d1853", stanzanlp)

# Intransitive sentence: "The storm abated"
demo = compare_with_pmb("pmb-gold-english", r"\p00\d2069", stanzanlp)

# Adverbial phrase: "Tom laughs a lot"
# demo = compare_with_pmb("pmb-gold-english", r"\p00\d2218", stanzanlp)
# (has a bug in PMB portion)

# Adverb: "Tom draws well"
demo = compare_with_pmb("pmb-gold-english", r"\p00\d2370", stanzanlp)

# Quantifier: "Klava oversimplifies everything"
# demo = compare_with_pmb("pmb-gold-english", r"\p00\d2540", stanzanlp)
# (has a bug in PMB portion)

# Control: "Tom started walking"
demo = compare_with_pmb("pmb-gold-english", r"\p00\d3051", stanzanlp)

# Prepositional phrase: "Pierce lives near Rossville Blvd"
demo = compare_with_pmb("pmb-gold-english", r"\p00\d0761", stanzanlp)
