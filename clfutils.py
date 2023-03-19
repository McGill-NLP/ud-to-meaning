from nltk.sem.drt import *
from sdrt import *
import re # helps in simplify_clf

argrelations = ("Agent","Theme","Patient","Topic","Recipient","Experiencer","Co_Theme","Co_Agent","Co-Theme","Co-Agent","Stimulus", "Creator", "User", "Of")

# Helps in converting NLTK DRS data structures to lists of clauses for CLF format.
# It takes as input a condition from the DRT module, the name of the projective DRS it belongs to,
# and a counter saying what names are still available for future DRS boxes.
# It returns a list of clauses and an updated counter.
def process_cond(cond, mydrsname, counter):
    if isinstance(cond,SdrtBoxRelationExpression):
        rel = cond.relation
        embdrs = cond.drs
        embname = 'b'+str(counter)
        newline = f'{mydrsname} {rel} {embname}'
        newlines, counter = drs_to_clf(embdrs,counter)
        return [newline] + newlines, counter
    elif isinstance(cond,DrtApplicationExpression):
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
        embname = 'b'+str(counter)
        newline = f'{mydrsname} NEGATION {embname}'
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
        embname2 = 'b'+str(counter)
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
    if not isinstance(drs,DRS):
        return [] if counter==0 else ([],counter)
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
    try:
        currentindex = 0
        currentbox = f'b{currentindex}'
        currentvariable = 0
        currentvariablename = f'x{currentvariable}'
        commentsplit = [x.split("%") for x in sbnlines]
        nonemptylines = [x[0].strip() for x in commentsplit if len(x)>0]
        nonemptylines = [x for x in nonemptylines if len(x)>0]
        variables = []
        boxrels = [[currentbox,'']] # unlike variables, we start out with a box there
        for line in nonemptylines:
            if line.split()[0].isupper():
                currentindex += 1
                currentbox = f'b{currentindex}'
                boxrels.append([currentbox, line])
            else:
                variables.append([currentbox, currentvariablename, line])
                currentvariable += 1
                currentvariablename = f'x{currentvariable}'
        boxes = [x for x,y in boxrels]
        clflines = [f"{var[0]} REF {var[1]}" for var in variables]
        for i in range(len(boxrels)):
            if boxrels[i][1]: # skip the first one, with an empty relation
                box = boxrels[i][0]
                rel, argpos = boxrels[i][1].split()
                otherboxind = i + ((-1 if (argpos.startswith('<') or argpos.startswith('-')) else 1)*int(argpos[1:]))
                otherbox = boxrels[otherboxind][0]
                clflines.append(f"{otherbox} {rel} {box}")
        # describe.v.01               Proposition <1
        for i in range(len(variables)):
            box, variable, predicates = variables[i]
            predlong = predicates.split(' ')[0]
            predshort, synset = predlong.split('.')[0], '.'.join(predlong.split('.')[1:])
            clflines.append(f"{box} {predshort} \"{synset}\" {variable}")

            relparts = [x for x in predicates.split(' ')[1:] if len(x)>0]
            relations_var = [(relparts[j],int(relparts[j+1])) for j in range(0,len(relparts)-1,2) if relparts[j+1][0] in ('+','-') and relparts[j+1][1:].isdigit()]
            relations_prop = [(relparts[j],(-1 if relparts[j+1].startswith('<') else 1)*int(relparts[j+1][1:])) for j in range(0,len(relparts)-1,2) if relparts[j+1][0] in ('<','>') and relparts[j+1][1:].isdigit()]
            relations_const = [(relparts[j],relparts[j+1]) for j in range(0,len(relparts)-1,2) if not (relparts[j+1][0] in ('+','-','<','>') and relparts[j+1][1:].isdigit())]
            relations_const = [(x,y[1:-1]) for x,y in relations_const if y.startswith('"') and y.endswith('"')] + [(x,y) for x,y in relations_const if not(y.startswith('"') and y.endswith('"'))] 

            clflines = clflines + [f"{box} {x} {variable} {variables[i+y][1]}" for x,y in relations_var]
            clflines = clflines + [f"{box} {x} {variable} \"{y}\"" for x,y in relations_const]
            clflines = clflines + [f"{box} {x} {variable} {boxes[boxes.index(box)+y]}" for x,y in relations_prop]
        return clflines
    except Exception as e:
        return []

def simplify_clf(clflines):
    clflinespctsplit = [x.split("%") for x in clflines]
    clflinestokens = [(x[0].split(),x[1:]) for x in clflinespctsplit]
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

def simplify_drs(drs):
    return drs

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

