from nltk.sem.drt import * # all the DRT things
from nltk.sem.logic import unique_variable # helps for manipulating DRT expressions
import graphviz # for displaying traces nicely
import conllu # reading ConLL-U files
import copy # deep-copying Tokens and and TokenLists
import logging
from semtypes import *
from conlluutils import *
from preprocessing import *
from sdrt import *

# All the word denotation templates, and relation denotations,
# are pairs of one SemType and one DRT expression (or template string for an expression)
# We read them from the files.
poslines = []
with open("postemplates.csv") as f:
    poslines = [x.split('\t') for x in f.read().split('\n') if len(x)>0]
postemplates = dict((x[0],[]) for x in poslines)
poswithnoden = []
for x in poslines:
    if x[1] == 'NA':
        poswithnoden.append(x[0])
    else:
        postemplates[x[0]].append((SemType.fromstring(x[1]),x[2]))

rellines = []
with open("reldenotations.csv") as f:
    rellines = [x.split('\t') for x in f.read().split('\n') if len(x)>0]
relmeanings = dict((x[0],[]) for x in rellines)
relswithnoden = []
for x in rellines:
    if x[1] == 'NA':
        relswithnoden.append(x[0])
    else:
        relmeanings[x[0]].append((SemType.fromstring(x[1]),SdrtExpression.fromstring(x[2])))

def add_denotation(t):
    '''
    Takes a Token (from conllu class) as input
    and returns a new Token
    identical to the first, but with denotations added
    for both the word itself ("word_den") and its relation ("rel_den").
    It adds no denotation to punctuation Tokens,
    and adds a denotation based on one of the templates to any non-determiner word.

            Parameters:
                    t (Token): The conllu-class Token to add the denotation to

            Returns:
                    t (Token): A new Token, just like the first, but with denotations added.
    '''
    t = copy.deepcopy(t)
    if t['upos'] in poswithnoden:
        if t['deprel'] == 'root':
            lemmastring = t['lemma']
            t['word_dens'] = [(template[0],SdrtExpression.fromstring(template[1].format(lemmastring)))
                            for template in postemplates['EXTRA']]
        else:
            return t
    elif t['upos'] in postemplates.keys():
        lemmastring = t['lemma']
        featurestring = ';'.join(';'.join((key,str(t['feats'][key]))) for key in t['feats'].keys()) if t['feats'] else ""
        t['word_dens'] = [(template[0],
                           (SdrtExpression.fromstring(template[1].format(featurestring,lemmastring)) if template[1].count('{')==2 else SdrtExpression.fromstring(template[1].format(lemmastring))))
                            for template in postemplates[t['upos']]]
    else:
        logging.warning("The word {} with ID {} is a POS with unknown denotation.".format(t['form'],str(t['id'])))
    if t['deprel'] in relswithnoden:
        return t
    elif t['deprel'] in relmeanings.keys():
        t['rel_dens'] = relmeanings[t['deprel']]
    elif ':' in t['deprel'] and t['deprel'].split(':')[0] in relmeanings.keys():
        t['rel_dens'] = relmeanings[t['deprel'].split(':')[0]]
    elif ':' in t['deprel'] and t['deprel'].split(':')[0] in relswithnoden:
        return t
    else:
        logging.warning("The relation {} on the word {} with ID {} is a relation with unknown denotation.".format(t['deprel'], t['form'], str(t['id'])))
    return t

def compute_conj(den1, den2, semtype):
    '''
    Computes the Lasersohnian conjunction of two lambda-expressions.
    Slightly broken; only guaranteed to work on the semantic types I've explicitly written in.

            Parameters:
                    den1: A DRS lambda expression (from NLTK) to conjoin
                    den2: Another DRS lambda expression (from NLTK) to conjoin
                    semtype: The shared semantic type of these expressions

            Returns:
                    conjden: A DRT expression representing the conjunction of the two lambda expressions, also with type semtype.
    '''
    # TODO this is broken now that there are new denotations for everything.
    # There's a special proviso for semantic types ((et)t) and ((st)t)
    if semtype.like(SemType.fromstring('((ut)t)')):
        conjden = SdrtExpression.fromstring(r'\F\G\H.(([x],[])+H(x)+F((\y.([],[Sub(y,x)]))) + G((\y.([],[Sub(y,x)]))))')
        return conjden(den1)(den2).simplify()
    # Deal with atomic types first
    if semtype == SemType.fromstring('t'):
        return den1 + den2
    if semtype.is_atomic():
        return SdrtExpression.fromstring(str(den1)+"-and-"+str(den2))
    # And a few other special cases
    if semtype.like(SemType.fromstring('(ut)')):
        conjden = SdrtExpression.fromstring(r'\F.\G.\x.(F(x)+G(x))')
        return conjden(den1)(den2).simplify()
    if semtype.like(SemType.fromstring('(u(st))')):
        conjden = SdrtExpression.fromstring(r'\F.\G.\x.\y.(F(x,y)+G(x,y))')
        return conjden(den1)(den2).simplify()
    if semtype.like(SemType.fromstring('(e(e(st)))')):
        conjden = SdrtExpression.fromstring(r'\F.\G.\x.\y.\z.(F(x,y,z)+G(x,y,z))')
        return conjden(den1)(den2).simplify()
    logging.warning(f"Not currently able to handle conjunctions of things of type {semtype}")
    # Now we (attempt to) deal with types that are functions from somewhere to somewhere else.
    # It will involve heavy use of new variables.
    # This portion might be have bugs currently.
    lefttype = semtype.get_left()
    if lefttype.is_atomic() and lefttype != SemType.fromstring('t'):
        a = DrtVariableExpression(unique_variable())
        astr = str(a)
        b = DrtVariableExpression(unique_variable())
        bstr = str(b)
        x = DrtVariableExpression(unique_variable())
        xstr = str(x)
        recursivecallstr = str(compute_conj(den1(a).simplify(),den2(b).simplify(),semtype.get_right()))
        return SdrtExpression.fromstring(rf'\{xstr}.({recursivecallstr}+([{astr} {bstr}],[Sub({astr},{xstr}) Sub({bstr},{xstr})]))').simplify()
    else:
        x = DrtVariableExpression(unique_variable())
        xstr = str(x)
        recursivecallstr = str(compute_conj(den1(x).simplify(),den2(x).simplify(),semtype.get_right()))
        return SdrtExpression.fromstring(rf'\{xstr}.{recursivecallstr}').simplify()

def semtypebinarizations(start, end, iopairs, comp_func = lambda x, y: x==y):
    '''
    The function returns a list of lists;
    each list in the returned list contains an ordering of the numbers 0 through len(iopairs)-1
    such that if you order iopairs in this order,
    then the first iopair's first component is start,
    the last iopair's second component is end,
    and otherwise each iopair's second component is the next iopair's first component.
    Like a valid Domino train.
    The function finds all possible such orderings (which might be 0) through depth-first search.
    This is used to find a correct order of composition for a Token's dependents.

            Parameters:
                    start: any type of thing, but usually a SemType
                    end: any type of thing, but usually a SemType
                    iopairs: a list of tuples of things of the same type as start and end
                    comp_func: optional. A function that takes two things of the types of start and end, and returns a boolean which should be taken to express whether those are equal.
            Returns:
                    binarizations: A list of lists representing the valid orderings of iopairs.
    '''
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

def multidominobinarizations(start, end, iopairs, comp_func = lambda x, y: x==y):
    '''
    The function returns a list of lists;
    each list in the returned list contains 2-uples.
    Their first elements of the 2-uples give an ordering of the numbers 0 through len(iopairs)-1
    by the same principles as semtypebinarizations.
    But now we are giving it an option of one of several iopairs for each ordering-element
    (like dominos with multiple faces)
    so the second elements of the 2-uples tell which element of iopairs is chosen for that domino.
    This is used to find a correct order of composition for a Token's dependents,
    when multiple semantic types are possible.

            Parameters:
                    start: any type of thing, but usually a SemType
                    end: any type of thing, but usually a SemType
                    iopairs: a list of lists of tuples of things of the same type as start and end
                    comp_func: optional. A function that takes two things of the types of start and end, and returns a boolean which should be taken to express whether those are equal.
            Returns:
                    binarizations: A list of lists representing the valid orderings of iopairs, with each ordering also telling you which "domino face" to use for each iopair.
    '''
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
 
def simplifynodetyped(treenode, withtrace=False):
    '''
    This function take a TokenTree as input,
    and returns a new TokenTree with no children
    It traverses the tree depth-first,
    and for each node's children, uses the binarization code
    to decide how best to compose the children with the node.
    Then, it uses a computed valid binarization to combine all the children into the node.
    It repeats until it finishes with the root, which it treats specially,
    because "root" relation has no head.
    If all the nodes in a TokenTree have been correctly typed and given denotations,
    applying simplifynodetyped to the root should give a correct denotation for the sentence
    in the "word_dens" field of the resulting TokenTree.
    If withtrace is True, then the denotations returned will have 3 components,
    with the third a dictionary
    containing "children" (a list of children in binarization order)
    and "original" (the original denotation of the node)

            Parameters:
                    treenode: A node from a Tree representation of a conllu TokenList
                    withtrace: Whether to return a list of denotations with a trace allowing you to replay the computation process
            Returns:
                    treenode: A node, though of as having no children, whose "word_dens" field contains all the possible denotations computed for the sentence.
    '''
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
    ends = [ends[i] for i in set([ends.index(x) for x in ends])]
    iopairs = [starts] + iopairs + [ends]
    binarizations = multidominobinarizations(SemType(),SemType(),iopairs,comp_func = lambda x,y:x.like(y))
    # Binarizations tell you which dependents to combine first.
    if binarizations:
        nodedens = []
        for binarization in binarizations:
            nodeden = treenode.token['word_dens'][binarization[0][1]]
            if withtrace:
                newnodedens = [(nodeden[0],nodeden[1],{'children':[],'original':nodeden,'form':treenode.token['form'],'deprel':treenode.token['deprel'],'upos':treenode.token['upos']})]
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
                                        {'children':newnodeden[2]['children'] + [(childrelden,childden)],'original':newnodeden[2]['original'],'form':newnodeden[2]['form'],'deprel':newnodeden[2]['deprel'],'upos':newnodeden[2]['upos']}
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
                                        {'children':newnodeden[2]['children'] + [(childrelden,childden)],'original':newnodeden[2]['original'],'form':newnodeden[2]['form'],'deprel':newnodeden[2]['deprel'],'upos':newnodeden[2]['upos']}
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
        logging.warning(f"There was a problem in binarizing children of node {treenode.token['id']} ({treenode.token['form']}, POS:{treenode.token['upos']})")
        logging.debug(f"Node {treenode.token['id']} ({treenode.token['form']}, POS: {treenode.token['upos']}, deprel: {treenode.token['deprel']}) children are related by unbinarizable relations {[child.token['deprel'] for child in usefulchildren]}")
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

def getalldens(tokenlist, withtrace = False):
    '''
    Turn a raw TokenList (from conllu package) into a list of all of the denotations computed for it.

            Parameters:
                    tokenlist: The TokenList representing the sentence to compute denotations for.
                    withtrace: If True, the list returned is a list of denotations, types, and traces. If False, the traces are left out.

            Returns:
                    dens: The list of denotations computed. Each denotation is a tuple of a DRTExpression, type, and potentially a trace.
    '''
    withdens = conllu.TokenList([add_denotation(token) for token in tokenlist])
    simplified = simplifynodetyped(withdens.to_tree(), withtrace)
    return simplified.token['word_dens'] if 'word_dens' in simplified.token.keys() else []

def tracetogvtree(t, counter = 0, graph = None, parentname = None):
    '''
    Turn a trace (as returned by simplifynodetyped) into a GraphViz directed graph
    which can then be viewed, or styled, or whatever as desired.

            Parameters:
                    t: A trace (as returned by simplifynodetyped)
                    counter: During recursive function calls, a counter to prevent nodes being assigned the same name.
                    graph: During recursive function calls, the parent graph.
                    parentname: During recursive function calls, the node name of the parent of the current node.

            Returns:
                    graph: a GraphViz directed graph representing the trace.
                    counter: During recursive function calls, a counter to prevent nodes being assigned the same name.
    '''
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


def tracetosemtypes(trace):
    '''
    # Get a list of the semantic types assigned to each relation and word in computing a trace.

            Parameters:
                    trace: A trace (as returned by simplifynodetyped) to convert.

            Returns:
                    lines: A list of strings representing pairs of one token or relation and the type assigned to it in that trace.
    '''
    if len(trace)==2:
        return []
    lines = []
    POS = trace[2]['upos']
    semtype = trace[2]['original'][0]
    lines.append((f'POS:{POS}',f'{semtype}'))
    for child in trace[2]['children']:
        deprel = child[1][2]['deprel']
        reltype = child[0][0]
        lines.append((f'deprel:{deprel}',f'{reltype}'))
        lines = lines + tracetosemtypes(child[1])
    return lines