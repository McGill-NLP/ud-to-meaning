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
def clf_to_drs(clfraw):
    clflines = clfraw.split("\n")
    clflinesstripped = [x.split("%")[0].strip() for x in clflines]
    contentlines = [x for x in clflinesstripped if len(x) > 0]
    pointedlines = {}
    pointerrelations = []
    for x in contentlines:
        if not (x[-2] == 'b' and x[-1].isdigit() and (not any(y.islower() for y in x[3:-3]))):
            if x[:2] in pointedlines.keys():
                pointedlines[x[:2]].append(x[3:])
            else:
                pointedlines[x[:2]] = [x[3:]]
        else:
            pointerrelations.append((x[:2],x[-2:],x[3:-3]))
    pointedrefs = {}
    pointedconds = {}
    for pointer in pointedlines.keys():
        pointedrefs[pointer] = []
        pointedconds[pointer] = []
        for statement in pointedlines[pointer]:
            if statement.startswith("REF "):
                pointedrefs[pointer].append(statement[4:])
            else:
                stmtsplit = statement.split(' ')
                argnames = [x for x in stmtsplit[1:] if not (len(x) > 2 and x[1].isalpha() and x[2] == '.' and x[3:-1].isdigit())]
                argnamesclean = [x.replace("+","more") for x in argnames]
                pointedconds[pointer].append(stmtsplit[0] + "(" + ",".join(argnamesclean) + ")")
    pointedboxes = {}
    for pointer in pointedlines.keys():
        pointedboxes[pointer] = DrtExpression.fromstring("([" + ",".join(pointedrefs[pointer]) + "],[" + ",".join(pointedconds[pointer]) + "])")
    neglectedpointers = [x[0] for x in pointerrelations if x[0] not in pointedlines.keys()] + [x[1] for x in pointerrelations if x[1] not in pointedlines.keys()]
    for pointer in neglectedpointers:
        pointedboxes[pointer] = DrtExpression.fromstring("([],[])")
    # For some reason, if one box presupposes another, they are written as one box.
    presuppositionrels = [x for x in pointerrelations if x[2] == "PRESUPPOSITION"]
    presuppositionrels.sort(key=lambda x:x[1])
    otherboxrels = [x for x in pointerrelations if x[2] != "PRESUPPOSITION"]
    presupcollapses = {}
    for relation in presuppositionrels:
        if relation[0] in pointedboxes.keys() and relation[1] in pointedboxes.keys():
            if relation[1] in presupcollapses.keys():
                presupcollapses[relation[1]].append(relation[0])
            else:
                presupcollapses[relation[1]] = [relation[0]]
            pointedboxes[relation[1]] = (pointedboxes[relation[1]] + pointedboxes[relation[0]]).simplify()
    # and update the relations we'll use to collapse boxes together too
    for newlabel in presupcollapses.keys():
        for oldlabel in presupcollapses[newlabel]:
            if oldlabel in pointedboxes.keys():
                del pointedboxes[oldlabel]
            for i in range(len(otherboxrels)):
                rel = otherboxrels[i]
                if rel[0] == oldlabel:
                    otherboxrels[i] = (newlabel,rel[1],rel[2])
                elif rel[1] == oldlabel:
                    otherboxrels[i] = (rel[0],newlabel,rel[2])
    # Treat other box-level relations as subordinating one box to another.
    otherboxrels.sort(key = lambda x:x[1], reverse=True)
    for relation in otherboxrels:
        if relation[0] in pointedboxes.keys() and relation[1] in pointedboxes.keys():
            pointedboxes[relation[0]] = pointedboxes[relation[0]] + DrtExpression.fromstring("(["+ relation[1] + "],[" + relation[2] + "(" + relation[1] + "), " + relation[1] + ":" + str(pointedboxes[relation[1]]) + "])")
            del pointedboxes[relation[1]]
    finaldrs = DrtExpression.fromstring("([],[])")
    for pointer in pointedboxes.keys():
        finaldrs = finaldrs + pointedboxes[pointer]
    return DrtExpression.fromstring(str(finaldrs.simplify()))

# This takes a raw clause format DRS (the type available in PMB data)
# and outputs a DRS of the class from the DRT module,
# but ALSO removes any extra information from proper nouns that I'm not able to get - 
# any line that came from the same word as another line's Name predicate,
# but itself is a lowercase predicate (like "male", etc.)
def clf_to_drs_no_propnouns(clfraw):
    clflines = clfraw.split("\n")
    clflinessplit = [x.split("%") for x in clflines]
    textwords = set(x[1] for x in clflinessplit if len(x)>1)
    linestodrop = []
    for word in textwords:
        sameword = [x for x in clflinessplit if (len(x)>1 and x[1] == word)]
        if sameword and max((x[0][3:7] == "Name") for x in sameword): # if this came from the same word
            for x in sameword:
                if len(x[0]) > 3 and x[0][3].islower():
                    linestodrop.append(x)
    clflinesstripped = [x[0].strip() for x in clflinessplit if x not in linestodrop]
    contentlines = [x for x in clflinesstripped if len(x) > 0]
    pointedlines = {}
    pointerrelations = []
    for x in contentlines:
        if not (x[-2] == 'b' and x[-1].isdigit() and (not any(y.islower() for y in x[3:-3]))):
            if x[:2] in pointedlines.keys():
                pointedlines[x[:2]].append(x[3:])
            else:
                pointedlines[x[:2]] = [x[3:]]
        else:
            pointerrelations.append((x[:2],x[-2:],x[3:-3]))
    pointedrefs = {}
    pointedconds = {}
    for pointer in pointedlines.keys():
        pointedrefs[pointer] = []
        pointedconds[pointer] = []
        for statement in pointedlines[pointer]:
            if statement.startswith("REF "):
                pointedrefs[pointer].append(statement[4:])
            else:
                stmtsplit = statement.split(' ')
                argnames = [x for x in stmtsplit[1:] if not (len(x) > 2 and x[1].isalpha() and x[2] == '.' and x[3:-1].isdigit())]
                argnamesclean = [x.replace("+","more") for x in argnames]
                pointedconds[pointer].append(stmtsplit[0] + "(" + ",".join(argnamesclean) + ")")
    pointedboxes = {}
    for pointer in pointedlines.keys():
        pointedboxes[pointer] = DrtExpression.fromstring("([" + ",".join(pointedrefs[pointer]) + "],[" + ",".join(pointedconds[pointer]) + "])")
    neglectedpointers = [x[0] for x in pointerrelations if x[0] not in pointedlines.keys()] + [x[1] for x in pointerrelations if x[1] not in pointedlines.keys()]
    for pointer in neglectedpointers:
        pointedboxes[pointer] = DrtExpression.fromstring("([],[])")
    # For some reason, if one box presupposes another, they are written as one box.
    presuppositionrels = [x for x in pointerrelations if x[2] == "PRESUPPOSITION"]
    presuppositionrels.sort(key=lambda x:x[1])
    otherboxrels = [x for x in pointerrelations if x[2] != "PRESUPPOSITION"]
    presupcollapses = {}
    for relation in presuppositionrels:
        if relation[0] in pointedboxes.keys() and relation[1] in pointedboxes.keys():
            if relation[1] in presupcollapses.keys():
                presupcollapses[relation[1]].append(relation[0])
            else:
                presupcollapses[relation[1]] = [relation[0]]
            pointedboxes[relation[1]] = (pointedboxes[relation[1]] + pointedboxes[relation[0]]).simplify()
    # and update the relations we'll use to collapse boxes together too
    for newlabel in presupcollapses.keys():
        for oldlabel in presupcollapses[newlabel]:
            if oldlabel in pointedboxes.keys():
                del pointedboxes[oldlabel]
            for i in range(len(otherboxrels)):
                rel = otherboxrels[i]
                if rel[0] == oldlabel:
                    otherboxrels[i] = (newlabel,rel[1],rel[2])
                elif rel[1] == oldlabel:
                    otherboxrels[i] = (rel[0],newlabel,rel[2])
    # Treat other box-level relations as subordinating one box to another.
    otherboxrels.sort(key = lambda x:x[1], reverse=True)
    for relation in otherboxrels:
        if relation[0] in pointedboxes.keys() and relation[1] in pointedboxes.keys():
            pointedboxes[relation[0]] = pointedboxes[relation[0]] + DrtExpression.fromstring("(["+ relation[1] + "],[" + relation[2] + "(" + relation[1] + "), " + relation[1] + ":" + str(pointedboxes[relation[1]]) + "])")
            del pointedboxes[relation[1]]
    finaldrs = DrtExpression.fromstring("([],[])")
    for pointer in pointedboxes.keys():
        finaldrs = finaldrs + pointedboxes[pointer]
    return DrtExpression.fromstring(str(finaldrs.simplify()))

# Removes tense
# (any variable which is the second argument in a Time(e,t) relation,
# and any conditions with that variable)
# and converts all theta roles to just Arg
def strip_tense_and_theta(drs):
    conds = [x for x in drs.conds]
    refs = drs.refs
    # recursive application to sub-DRS:
    for i in range(len(conds)):
        if isinstance(conds[i],DrtProposition):
            cond = conds[i]
            conds[i] = DrtProposition(cond.variable,strip_tense_and_theta(cond.drs))
    for i in range(len(conds)):
        if isinstance(conds[i], ApplicationExpression) and conds[i].is_atom(): # if it's a relation thing
            condstring = str(conds[i].pred)
            if condstring in ("Agent","Theme","Topic","Recipient","Experiencer"):
                newpred = DrtConstantExpression(Variable("Arg"))
                for arg in conds[i].args:
                    newpred = DrtApplicationExpression(newpred,arg)
                conds[i] = newpred
    newconds = conds
    newrefs = refs
    # tense:
    for cond in conds:
        if isinstance(cond,ApplicationExpression) and cond.is_atom():
            if str(cond.pred) == "Time":
                timevar = cond.argument.variable
                newrefs = [x for x in newrefs if x != timevar] # remove the time variable
                newconds = remove_conds_with_var(newconds,timevar) # and remove any condition that has it
        pass
    return DRS(newrefs,newconds)

# Recursively removes any conditions that have a particular variable in them.
def remove_conds_with_var(conds,var):
    newconds = []
    for cond in conds:
        if isinstance(cond,DrtProposition):
            if cond.variable != var and var not in [x.variable for x in cond.drs.refs]:
                newconds.append(DrtProposition(cond.variable,
                                        DRS(cond.drs.refs,remove_conds_with_var(cond.drs.conds,var))))
        elif not var in cond.variables():
            newconds.append(cond)
    return newconds

# Removes argument numbers - e.g. replaces Arg1 with just Arg
def strip_arg_nums(drs):
    conds = [x for x in drs.conds]
    # recursive application to sub-DRS:
    for i in range(len(conds)):
        if isinstance(conds[i],DrtProposition):
            cond = conds[i]
            conds[i] = DrtProposition(cond.variable,strip_arg_nums(cond.drs))
    refs = drs.refs
    # taking off numbers from Arg
    for i in range(len(conds)):
        if isinstance(conds[i], ApplicationExpression):
            if conds[i].is_atom(): # if it's a relation thing
                condstring = str(conds[i].pred)
                if len(condstring)==4 and condstring.startswith("Arg") and condstring[3].isdigit():
                    # replace the word Arg1, Arg2, etc. with just the word Arg.
                    newpred = DrtConstantExpression(Variable("Arg"))
                    for arg in conds[i].args:
                        newpred = DrtApplicationExpression(newpred,arg)
                    conds[i] = newpred
    return DRS(refs,conds)

# This function just makes demos easier.
def compare_with_pmb(pmbpath, datapointpath, stanzanlp, simplified=False):
    # Get the raw text, tokens, and gold DRT structure.
    with open(pmbpath + datapointpath + r"\en.raw") as f:
        datapointraw = f.read()
    print(datapointraw)

    with open(pmbpath + datapointpath + r"\en.tok.off") as f:
        tokensraw = f.read()
    tokens = ['~'.join(x.split(' ')[3:]) for x in tokensraw.split('\n') if len(x) > 0]

    with open(pmbpath + datapointpath + r"\en.drs.clf") as f:
        drsclfraw = f.read()

    pmb_drs = strip_tense_and_theta(clf_to_drs_no_propnouns(drsclfraw)) if simplified else clf_to_drs(drsclfraw)

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

    print("\nPMB DRS:\n"+pmb_drs.pretty_format()+"\n\nMy DRS:\n"+(strip_arg_nums(simpleparsing_drs[0][1]) if simplified else simpleparsing_drs[0][1]).pretty_format())

    return pmb_drs, simpleparsing_drs

# MARK here is where specific demos begin
# Basic transitive sentence: "Kraft sold Celestial Seasonings"
demo = compare_with_pmb("pmb-gold-english", r"\p00\d0712", stanzanlp)
demo = compare_with_pmb("pmb-gold-english", r"\p00\d0712", stanzanlp, simplified=True)

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
