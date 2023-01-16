#from nltk.inference import TableauProver # helps resolve anaphora
from nltk.sem.drt import * # all the DRT things
from nltk.sem.logic import unique_variable # helps for manipulating DRT expressions
import graphviz # for displaying traces nicely
from pptree import print_tree # helps print traces nicely
import conllu # reading ConLL-U files
import os # changing working directory
import copy # deep-copying Tokens and and TokenLists
# whatever the working directory is! on my computer it is this.
os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')
# the local modules and files should be in the working directory now
from semtypes import *
from conlluutils import *
from preprocessing import *

# All the word denotation templates, and relation denotations,
# are pairs of one SemType and one DRT expression (or template string for an expression)
# We read them from the files.
poslines = []
with open("postemplates.csv") as f:
    poslines = [x.split('\t') for x in f.read().split('\n')]
postemplates = dict((x[0],[]) for x in poslines)
poswithnoden = []
for x in poslines:
    if x[1] == 'NA':
        poswithnoden.append(x[0])
    else:
        postemplates[x[0]].append((SemType.fromstring(x[1]),x[2]))

rellines = []
with open("reldenotations.csv") as f:
    rellines = [x.split('\t') for x in f.read().split('\n')]
relmeanings = dict((x[0],[]) for x in rellines)
relswithnoden = []
for x in rellines:
    if x[1] == 'NA':
        relswithnoden.append(x[0])
    else:
        relmeanings[x[0]].append((SemType.fromstring(x[1]),DrtExpression.fromstring(x[2])))

# This function takes a Token as input
# and returns a new Token
# identical to the first, but with denotations added
# for both the word itself ("word_den") and its relation ("rel_den").
# It adds no denotation to punctuation Tokens,
# and adds a denotation based on one of the templates to any non-determiner word.
def add_denotation(t):
    t = copy.deepcopy(t)
    if t['upos'] in poswithnoden:
        return t
    elif t['upos'] in postemplates.keys():
        t['word_dens'] = [(template[0],DrtExpression.fromstring(template[1].format(t['lemma'].lower().replace("-","_").replace("exist","exst").replace(".","").replace("all","alll"))))
                            for template in postemplates[t['upos']]]
    else:
        print("The word {} with ID {} is a POS with unknown denotation.".format(t['form'],str(t['id'])))
    if t['deprel'] in relswithnoden:
        return t
    elif t['deprel'] in relmeanings.keys():
        t['rel_dens'] = relmeanings[t['deprel']]
    elif ':' in t['deprel'] and t['deprel'].split(':')[0] in relmeanings.keys():
        t['rel_dens'] = relmeanings[t['deprel'].split(':')[0]]
    elif ':' in t['deprel'] and t['deprel'].split(':')[0] in relswithnoden:
        return t
    else:
        print("The relation {} on the word {} with ID {} is a relation with unknown denotation.".format(t['deprel'], t['form'], str(t['id'])))
    return t

# This function computes the conjunction (in an essentially Lasersohn way) of two lambda-expressions
# den1 and den2 are two DRS lambda-expressions.
# They must both have the same semantic type, which is semtype.
# The function computes and returns their conjunction,
# which has the same semantic type.
# It won't work very well until we fix the DRS simplification code.
# TODO this is broken now that there are new denotations for everything.
def compute_conj(den1, den2, semtype):
    # There's a special proviso for semantic types ((et)t) and ((st)t)
    if semtype.like(SemType.fromstring('((ut)t)')):
        conjden = DrtExpression.fromstring(r'\F\G\H.(([x],[])+H(x)+F((\y.([],[Sub(y,x)]))) + G((\y.([],[Sub(y,x)]))))')
        return conjden(den1)(den2).simplify()
    # Deal with atomic types first
    if semtype == SemType.fromstring('t'):
        return den1 + den2
    elif semtype.is_atomic():
        return DrtExpression.fromstring(str(den1)+"-and-"+str(den2))
    # Now we deal with types that are functions from somewhere to somewhere else.
    # It will involve heavy use of new variables.
    lefttype = semtype.get_left()
    if lefttype.is_atomic() and lefttype != SemType.fromstring('t'):
        a = DrtVariableExpression(unique_variable())
        astr = str(a)
        b = DrtVariableExpression(unique_variable())
        bstr = str(b)
        x = DrtVariableExpression(unique_variable())
        xstr = str(x)
        recursivecallstr = str(compute_conj(den1(a).simplify(),den2(b).simplify(),semtype.get_right()))
        return DrtExpression.fromstring(rf'\{xstr}.({recursivecallstr}+([{astr} {bstr}],[Sub({astr},{xstr}) Sub({bstr},{xstr})]))').simplify()
    else:
        x = DrtVariableExpression(unique_variable())
        xstr = str(x)
        recursivecallstr = str(compute_conj(den1(x).simplify(),den2(x).simplify(),semtype.get_right()))
        return DrtExpression.fromstring(rf'\{xstr}.{recursivecallstr}').simplify()

# This function takes four inputs:
# start is any type of thing, but usually a SemType
# end is any type of thing, but usually a SemType
# iopairs is a list of tuples of things of the same type as start and end
# comp_func is an optional argument. It is a function that takes two things of the types of start and end,
#             and returns a boolean which should be taken to express whether those are equal.
# The function returns a list of lists;
# each list in the returned list contains an ordering of the numbers 0 through len(iopairs)-1
# such that if you order iopairs in this order,
# then the first iopair's first component is start,
# the last iopair's second component is end,
# and otherwise each iopair's second component is the next iopair's first component.
# Like a valid Domino train.
# The function finds all possible such orderings (which might be 0) through depth-first search.
# This is used to find a correct order of composition for a Token's dependents.
def semtypebinarizations(start, end, iopairs, comp_func = lambda x, y: x==y):
    binarizations = []
    if len(iopairs)==1:
        if comp_func(iopairs[0][0],start) and comp_func(iopairs[0][1],end):
            binarizations.append([0])
    else:
        for i in range(len(iopairs)):
            pair = iopairs[i]
            if comp_func(pair[0],start):
                tails = semtypebinarizations(pair[1],end,iopairs[:i]+iopairs[i+1:],comp_func)
                for tail in tails:
                    for j in range(len(tail)):
                        if tail[j] >= i:
                            tail[j] = tail[j] + 1
                    binarizations.append([i]+tail)
    return binarizations

# This function takes four inputs:
# start is any type of thing, but usually a SemType
# end is any type of thing, but usually a SemType
# iopairs is a list of lists of tuples of things of the same type as start and end
# comp_func is an optional argument. It is a function that takes two things of the types of start and end,
#             and returns a boolean which should be taken to express whether those are equal.
# The function returns a list of lists;
# each list in the returned list contains 2-uples.
# Their first elements of the 2-uples give an ordering of the numbers 0 through len(iopairs)-1
# by the same principles as semtypebinarizations.
# But now we are giving it an option of one of several iopairs for each ordering-element
# (like dominos with multiple faces)
# so the second elements of the 2-uples tell which element of iopairs is chosen for that domino.
# This is used to find a correct order of composition for a Token's dependents,
# when multiple semantic types are possible
# TODO this is unfortunately very slow now
def multidominobinarizations(start, end, iopairs, comp_func = lambda x, y: x==y):
    binarizations = []
    if len(iopairs)==1:
        for k in range(len(iopairs[0])):
            if comp_func(iopairs[0][k][0],start) and comp_func(iopairs[0][k][1],end):
                binarizations.append([(0,k)])
    else:
        for m in range(len(iopairs)):
            pair = iopairs[m]
            for k in range(len(pair)):
                if comp_func(pair[k][0],start):
                    tails = multidominobinarizations(pair[k][1],end,iopairs[:m]+iopairs[m+1:],comp_func)
                    for tail in tails:
                        for n in range(len(tail)):
                            if tail[n][0] >= m:
                                tail[n] = (tail[n][0] + 1, tail[n][1])
                        binarizations.append([(m,k)]+tail)
    return binarizations
 
# This function take a TokenTree as input,
# and returns a new TokenTree with no children
# It traverses the tree depth-first,
# and for each node's children, uses the binarization code
# to decide how best to compose the children with the node.
# Then, it uses a computed valid binarization to combine all the children into the node.
# It repeats until it finishes with the root, which it treats specially,
# because "root" relation has no head.
# If all the nodes in a TokenTree have been correctly typed and given denotations,
# applying simplifynodetyped to the root should give a correct denotation for the sentence
# in the "word_dens" field of the resulting TokenTree.
# If withtrace is True, then the denotations returned will have 3 components,
# with the third a dictionary
# containing "children" (a list of children in binarization order)
# and "original" (the original denotation of the node)
def simplifynodetyped(treenode, withtrace=False):
    # Ignore nodes whose types we don't know.
    treenode = copy.deepcopy(treenode)
    if ('rel_dens' not in treenode.token.keys()) or ('word_dens' not in treenode.token.keys()):
        return treenode
    # Depth-first: simplify children first.
    treenode.children = [simplifynodetyped(child, withtrace) for child in treenode.children]
    # Compute binarizations
    iopairs = []
    usefulchildren = []
    for child in treenode.children:
        if 'rel_dens' in child.token.keys() and 'word_dens' in child.token.keys():
            iopairs.append([(den[0].get_left(),den[0].get_right().get_right()) for den in child.token['rel_dens']])
            usefulchildren.append(child)
    # we'll encode multiple possible denotations of this node as possible first dominos
    # with dummy starting type
    starts = [(SemType(),den[0]) for den in treenode.token['word_dens']]
    ends = [(den[0].get_right().get_left(),SemType()) for den in treenode.token['rel_dens']]
    iopairs = [starts] + iopairs + [ends]
    binarizations = multidominobinarizations(SemType(),SemType(),iopairs,comp_func = lambda x,y:x.like(y))
    # Binarizations tell you which dependents to combine first.
    if binarizations:
        # TODO currently this leaves me with too many binarizations,
        # which are not genuinely different.
        # Need a way to tell whether two computed denotations are really different.
        nodedens = []
        for binarization in binarizations:
            nodeden = treenode.token['word_dens'][binarization[0][1]]
            if withtrace:
                newnodedens = [(nodeden[0],nodeden[1],{'children':[],'original':nodeden,'form':treenode.token['form'],'deprel':treenode.token['deprel']})]
            else:
                newnodedens = [(nodeden)]
            for term in binarization[1:-1]:
                child = usefulchildren[term[0]-1]
                childrelden = child.token['rel_dens'][term[1]]
                usefulchilddens = [den
                                    for den in child.token['word_dens']
                                    if den[0].like(childrelden[0].get_right().get_left())]
                if child.token['deprel'] == 'conj':
                    if withtrace:
                        newnodedens = [(childden[0],
                                        compute_conj(newnodeden[1],childden[1],childden[0]),
                                        {'children':newnodeden[2]['children'] + [(childrelden,childden)],'original':newnodeden[2]['original'],'form':newnodeden[2]['form'],'deprel':newnodeden[2]['deprel']}
                                        )
                                                            for childden in usefulchilddens
                                                            for newnodeden in newnodedens]
                    else:
                        newnodedens = [(childden[0],compute_conj(newnodeden[1],childden[1],childden[0]))
                                                            for childden in usefulchilddens
                                                            for newnodeden in newnodedens]
                else:
                    if withtrace:
                        newnodedens = [(childrelden[0].get_right().get_right(),
                                                        childrelden[1](newnodeden[1])(childden[1]).simplify(),
                                        {'children':newnodeden[2]['children'] + [(childrelden,childden)],'original':newnodeden[2]['original'],'form':newnodeden[2]['form'],'deprel':newnodeden[2]['deprel']}
                                        )
                                                            for childden in usefulchilddens
                                                            for newnodeden in newnodedens]
                    else:
                        newnodedens = [(childrelden[0].get_right().get_right(),
                                                        childrelden[1](newnodeden[1])(childden[1]).simplify())
                                                            for childden in usefulchilddens
                                                            for newnodeden in newnodedens]
            nodedens = nodedens + newnodedens
        treenode.token['word_dens'] = nodedens
    elif iopairs:
        print("There was a problem in binarizing children of node {} ({})".format(treenode.token['id'], treenode.token['form']))
    # If the node is one that's conjoined to another node,
    # we want the "conj" type to enforce that the other node has the same semantic type,
    # so we have to update its semantic type after simplifying this node.
    # It will change the "conj" from "(?(??))" to replace all the ?'s with the type of this node.
    if treenode.token['deprel'] == 'conj':
        treenode.token['rel_dens'] = [(CompositeType(den[0],
                                        CompositeType(den[0],
                                        den[0])),None) for den in treenode.token['word_dens']]
    # Root only takes one argument
    if treenode.token['deprel'] == 'root':
        if withtrace:
            treenode.token['word_dens'] = [(relden[0].get_right().get_right(),
                                                    relden[1](wordden[1]).simplify(),
                                                    wordden[2])
                                                    for relden in treenode.token['rel_dens']
                                                    for wordden in treenode.token['word_dens']
                                                    if relden[0].get_right().get_left().like(wordden[0])]
        else:
            treenode.token['word_dens'] = [(relden[0].get_right().get_right(),
                                                    relden[1](wordden[1]).simplify())
                                                    for relden in treenode.token['rel_dens']
                                                    for wordden in treenode.token['word_dens']
                                                    if relden[0].get_right().get_left().like(wordden[0])]
    return treenode

# pass in a raw conllu or tokenlist
# and get all of the denotations (or traces) computed for it
def getalldens(tokenlist, withtrace = False):
    withdens = conllu.TokenList([add_denotation(token) for token in tokenlist])
    simplified = simplifynodetyped(withdens.to_tree(), withtrace)
    return simplified.token['word_dens']

# Turn a trace (as returned by simplifynodetyped) into a GraphViz directed graph
# which can then be viewed, or styled, or whatever as desired.
# The arguments counter, graph, and parentname are mainly for use within the recursive function calls.
def tracetogvtree(t, counter = 0, graph = None, parentname = None):
    if graph is None:
        graph = graphviz.Digraph(node_attr={'shape': 'box','fontname':'Courier New'}, edge_attr={'dir':'back'})
    topname = str(counter)
    counter += 1
    if len(t) < 3:
        graph.node(topname, label=str(t))
    elif len(t[2]['children']) == 0:
        graph.node(topname, label = r'\n'.join([t[2]['form'], t[0].tuple_shaped_str(), t[1].pretty_format()]))
    else:
        graph.node(topname, label = r'\n'.join([t[2]['form'], "(final)", t[0].tuple_shaped_str(), t[1].pretty_format()]))
        prevname = topname
        for child in reversed(t[2]['children']):
            childnodeden = child[1]
            childrelden = child[0]
            nextname = str(counter)
            counter += 1
            graph.node(nextname, shape='plaintext', label = childrelden[0].get_right().get_right().tuple_shaped_str())
            graph.edge(prevname, nextname)
            prevname = nextname
            _, counter = tracetogvtree(childnodeden, counter, graph, nextname)
        bottomname = str(counter)
        counter += 1
        graph.node(bottomname, label = r'\n'.join([t[2]['form'],'(original)',t[2]['original'][0].tuple_shaped_str(),t[2]['original'][1].pretty_format()]))
        graph.edge(nextname,bottomname)
    if parentname is not None:
        graph.edge(parentname, topname, label = (t[2]['deprel'] if len(t) > 2 else None))
        return graph, counter
    else:
        return graph
