from nltk.sem.drt import *
import re # helps in simplify_clf

argrelations = ("Agent","Theme","Patient","Topic","Recipient","Experiencer","Co_Theme","Co_Agent","Co-Theme","Co-Agent","Stimulus", "Creator", "User", "Of")

# Helps in converting NLTK DRS data structures to lists of clauses for CLF format.
# It takes as input a condition from the DRT module, the name of the projective DRS it belongs to,
# and a counter saying what names are still available for future DRS boxes.
# It returns a list of clauses and an updated counter.
def process_cond(cond, mydrsname, counter):
    if isinstance(cond,DrtApplicationExpression):
        function, args = cond.uncurry()
        return([f'{mydrsname} {function} ' + ' '.join(str(x) for x in args)], counter)
    elif isinstance(cond,DrtProposition):
        var = cond.variable
        embdrs = cond.drs
        embname = 'b'+str(counter)
        newline = f'{mydrsname} PRP {var} {embname}'
        newlines, counter = drs_to_clf(embdrs,counter)
        return [newline] + newlines, counter
    elif isinstance(cond,DRS):
        return drs_to_clf(cond,counter)
    elif isinstance(cond,DrtNegatedExpression):
        embterm = cond.term
        embname = 'b'.str(counter)
        newline = f'{mydrsname} NOT {embname}'
        if not isinstance(embterm,DRS):
            counter = counter + 1
        newlines, counter = process_cond(embterm,embname,counter)
        return [newline] + newlines, counter
    elif isinstance(cond,DrtOrExpression):
        embterm1 = cond.first
        embname1 = 'b'+str(counter)
        if not isinstance(embterm1,DRS):
            counter = counter + 1
        newlines1, counter = process_cond(embterm1,embname1,counter)
        embterm2 = cond.second
        embname2 = 'b'+str(counter)
        if not isinstance(embterm2,DRS):
            counter = counter + 1
        newlines2, counter = process_cond(embterm2,embname2,counter)
        newline = f'{mydrsname} DIS {embname1} {embname2}'
        return [newline] + newlines1 + newlines2, counter
    elif isinstance(cond,DrtConcatenation):
        embterm1 = cond.first
        embname1 = 'b'+str(counter)
        if not isinstance(embterm1,DRS):
            counter = counter + 1
        newlines1, counter = process_cond(embterm1,embname1,counter)
        embterm2 = cond.second
        if not isinstance(embterm2,DRS):
            counter = counter + 1
        newlines2, counter = process_cond(embterm2,embname2,counter)
        newline = f'{mydrsname} CNJ {embname1} {embname2}'
        return [newline] + newlines1 + newlines2, counter
    elif isinstance(cond,DrtEqualityExpression):
        return [f'{mydrsname} EQU {cond.first} {cond.second}'], counter
    elif isinstance(cond,DrtBinaryExpression):
        embterm1 = cond.first
        embname1 = 'b'+str(counter)
        if not isinstance(embterm1,DRS):
            counter = counter + 1
        newlines1, counter = process_cond(embterm1,embname1,counter)
        embterm2 = cond.second
        if not isinstance(embterm2,DRS):
            counter = counter + 1
        newlines2, counter = process_cond(embterm2,embname2,counter)
        newline = f'{mydrsname} {cond.getOp()} {embname1} {embname2}'
        return  [newline] + newlines1 + newlines2, counter

# Converting NLTK DRS data structures to lists of clauses for CLF format.
# It takes as input a DRS from the DRT module,
# and a counter saying what names are still available for future DRS boxes.
# It returns a list of clauses.
# If it is being recursively called (or is called with a high counter number)
# it also returns an updated counter
def drs_to_clf(drs, counter=0):
    istop = counter==0
    mydrsname = 'b'+str(counter)
    counter = counter + 1
    clflines = []
    for x in drs.refs:
        clflines.append(mydrsname + " REF " + str(x))
    for cond in drs.conds:
        newlines, counter = process_cond(cond,mydrsname,counter)
        clflines = clflines + newlines
    if istop:
        return clflines
    return clflines, counter

def drses_to_clf(drslist):
    clflineses = [drs_to_clf(x) for x in drslist if isinstance(x,DRS)]
    return clflineses

# Converting Simplified Box Notation to CLF format.
def sbn_to_clf(sbnlines):
    commentsplit = [x.split("%") for x in sbnlines]
    nonemptylines = [x[0].strip() for x in commentsplit if len(x)>0]
    nonemptylines = [x for x in nonemptylines if len(x)>0]
    variables = ["x"+str(i) for i in range(1,len(nonemptylines)+1)]
    clflines = [f"b0 REF {x}" for x in variables]
    for i in range(len(nonemptylines)):
        predlong = nonemptylines[i].split(' ')[0]
        predshort, synset = predlong.split('.')[0], '.'.join(predlong.split('.')[1:])
        clflines.append(f"b0 {predshort} \"{synset}\" {variables[i]}")
        relparts = [x for x in nonemptylines[i].split(' ')[1:] if len(x)>0]
        print(relparts)
        relations_var = [(relparts[j],int(relparts[j+1])) for j in range(0,len(relparts)-1,2) if relparts[j+1][0] in ('+','-') and relparts[j+1][1:].isdigit()]
        relations_const = [(relparts[j],relparts[j+1]) for j in range(0,len(relparts)-1,2) if not (relparts[j+1][0] in ('+','-') and relparts[j+1][1:].isdigit())]
        clflines = clflines + [f"b0 {x} {variables[i]} {variables[i+y]}" for x,y in relations_var]
        clflines = clflines + [f"b0 {x} {variables[i]} \"{y}\"" for x,y in relations_const]
    return clflines

# This function takes a list of lines in CLF format
# and simplifies them by removing the information I can't get from them:
# extra information that comes with names,
# tense information,
# and theta roles.
def simplify_clf(clflines):
    clflinespctsplit = [x.split("%") for x in clflines]
    textwords = set(x[1] for x in clflinespctsplit if len(x)>1)
    # Remove lowercase lines that come from the same word as a Name line.
    if textwords:
        linestodrop = []
        for word in textwords:
            sameword = [x for x in clflinespctsplit if (len(x)>1 and x[1] == word)]
            if sameword and max((x[0][3:7] == "Name") for x in sameword): # if this came from the same word
                for x in sameword:
                    if len(x[0]) > 3 and x[0][3].islower():
                        linestodrop.append(x)
        clflinespctsplit = [x for x in clflinespctsplit if x not in linestodrop and len(x) > 0]
    clflinestokens = [(x[0].split(),x[1:]) for x in clflinespctsplit]
    # Remove the first argument on the line if it's a "sense" disambiguation thing
    for i in range(len(clflinestokens)):
        x, y = clflinestokens[i]
        if len(x) > 2:
            if len(x[2]) > 2 and x[2][0]=='"' and x[2][1].isalpha() and x[2][2] == '.' and x[2][3:-1].isdigit() and x[2][-1] == '"':
                clflinestokens[i] = (x[:2] + x[3:],y)
    # Change any ArgN or theta-roles to just Arg.
    for x,_ in clflinestokens:
        if len(x) > 1 and ((x[1] in argrelations) or (
                x[1].startswith('Arg') and x[1][3:].isdigit())):
            x[1] = "Arg"
    # Remove lines that include variable t if t participates in a Time relation as the second argument.
    timevars = [x[-1] for x,y in clflinestokens if len(x)>1 and x[1]=='Time']
    for t in timevars:
        clflinestokens = [(x,y) for x,y in clflinestokens if t not in x[2:]]
    # Remove box-level relations.
    clflinestokens = [(x,y) for x,y in clflinestokens if sum([re.match(r'^b\d*$',tok) is not None for tok in x if isinstance(tok,str)]) <= 1]
    # Make all box labels the same
    for x,_ in clflinestokens:
        if len(x) > 0 and re.match(r'^b\d*$',x[0]):
            x[0] = 'b0'
    return(['%'.join([' '.join(line[0])+' '] + line[1]) for line in clflinestokens])


# This takes list of the lines from a clause format CLF file (the type available in PMB data)
# and outputs a DRS of the class from the DRT module.
def clf_to_drs(clflines):
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
                pointedconds[pointer].append(stmtsplit[0].replace('-','_') + "(" + ",".join(argnamesclean) + ")")
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
def simplify_drs(drs):
    conds = [x for x in drs.conds]
    refs = drs.refs
    # recursive application to sub-DRS:
    for i in range(len(conds)):
        if isinstance(conds[i],DrtProposition):
            cond = conds[i]
            conds[i] = DrtProposition(cond.variable,simplify_drs(cond.drs))
    for i in range(len(conds)):
        if isinstance(conds[i], ApplicationExpression) and conds[i].is_atom(): # if it's a relation thing
            condstring = str(conds[i].pred)
            if condstring in argrelations or condstring.startswith("Arg"):
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

