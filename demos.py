import os
import stanza # parsing to UD
import conllu # for reading conllu files
#os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')
import conlluutils
import preprocessing
import treetodrs
import clfutils

stanzapipeline = None
def get_stanza():
    global stanzapipeline
    if stanzapipeline is None:
        stanzapipeline = stanza.Pipeline(lang='en', processors = 'tokenize,pos,lemma,depparse')
    return stanzapipeline

def pmbdemo(pmbpath, partnum, docnum, simplified=False):
    # Read the PMB CLF file,
    # (making sure input part number and document number are the right format)
    if isinstance(partnum,int):
        partnum = f'p{partnum:02d}'
    else:
        if not isinstance(partnum, str):
            partnum = str(partnum)
        if not partnum.startswith('p'):
            partnum = 'p'+partnum
    if isinstance(docnum,int):
        docnum = f'd{docnum:04d}'
    else:
        if not isinstance(docnum, str):
            docnum = str(docnum)
        if not docnum.startswith('d'):
            docnum = 'd'+docnum
    with open(pmbpath + "\\" + str(partnum) + "\\" + str(docnum) + r"\en.drs.clf") as f:
        pmbclfraw = f.read()
        pmbclflines = pmbclfraw.split('\n')
    if simplified:
        pmbclflines = clfutils.simplify_clf(pmbclflines)
    pmb_drs = clfutils.clf_to_drs(pmbclflines)
    # Read the raw PMB file,
    with open(pmbpath + "\\" + str(partnum) + "\\" + str(docnum) + r"\en.raw") as f:
        textraw = f.read()
    # Parse with Stanza,
    stanzanlp = get_stanza()
    ud_parse = conlluutils.stanza_to_conllu(stanzanlp(textraw))
    # Preprocess,
    preprocessed = preprocessing.preprocess(ud_parse)
    # Compute the meaning,
    ud_drss = treetodrs.getalldens(preprocessed)
    # Simplify if necessary,
    if ud_drss:
        if simplified:
            ans_drs = clfutils.simplify_drs(ud_drss[0][1])
        else:
            ans_drs = ud_drss[0][1]
        # Return all the outputs.
        return ud_parse, preprocessed, ans_drs, pmb_drs
    else:
        print("This one didn't produce any DRS output.")
        return ud_parse, preprocessed, None, pmb_drs

def conlludemo(conllupath):
    # Read the conllu,
    with open(conllupath) as f:
        conlluraw = f.read()
    conlluparse = conllu.parse(conlluraw)[0]
    # Preprocess,
    preprocessed = preprocessing.preprocess(conlluparse)
    # Compute the meaning,
    ud_drss = treetodrs.getalldens(preprocessed)
    # Return all the outputs.
    if ud_drss:
        return conlluparse, preprocessed, ud_drss[0][1]
    else:
        print("This one didn't produce any DRS output.")
        return conlluparse, preprocessed

def sentencedemo(text):
    # Parse with Stanza,
    stanzanlp = get_stanza()
    ud_parse = conlluutils.stanza_to_conllu(stanzanlp(text))
    # Preprocess,
    preprocessed = preprocessing.preprocess(ud_parse)
    # Compute the meaning,
    ud_drss = treetodrs.getalldens(preprocessed)
    # Return all the outputs.
    if ud_drss:
        return ud_parse, preprocessed, ud_drss[0][1]
    else:
        print("This one didn't produce any DRS output.")
        return ud_parse, preprocessed


if __name__ == "__main__":
    # TODO this command line tool
    # has a command line option for pmb, conllu, or sentence
    # takes the inputs it looks like we should take
    #   so, pmbpath, part, and doc number, and optionally "simplified"
    #   a sentence OR
    #   a path to a conll-u
    # prints a UD tree, a preprocessed UD tree, a DRS, and (if PMB) the original DRS.
    # TODO add options about traces in the future.
    print("I promise I'll implement this :)")