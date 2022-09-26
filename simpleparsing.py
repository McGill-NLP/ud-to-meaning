#from nltk.inference import TableauProver # helps resolve anaphora
from nltk.sem.drt import * # all the DRT things
from nltk.sem.logic import unique_variable # helps for manipulating DRT expressions
import conllu # reading ConLL-U files
import os # changing working directory
import copy # deep-copying Tokens and and TokenLists
# whatever the working directory is! on my computer it is this.
os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')
# the semtypes module should be in the working directory now
from semtypes import *
# (also the folder "conllus" for the demos later)

# MARK Functions to facilitate manipulating tokens and tokenlists

# This function takes a TokenTree as input,
# and returns a new TokenTree with most data the same as the original TokenTree,
# with two differences:
# first, the new TokenTree now has no children,
# and second, the "form" and "lemma" of the TokenTree's token are now
# concatenations of the existing "form" and "lemma" with those of all children
# and their descendants as well.
def concat_with_all_dependents(treenode):
    treenode = copy.deepcopy(treenode)
    for child in treenode.children:
        child = concat_with_all_dependents(child)
        if child.token['id'] < treenode.token['id']:
            treenode.token['form'] = child.token['form'] + '_' + treenode.token['form']
            treenode.token['lemma'] = child.token['lemma'] + '_' + treenode.token['lemma']
        else:
            treenode.token['form'] = treenode.token['form'] + '_' + child.token['form']
            treenode.token['lemma'] = treenode.token['lemma'] + '_' + child.token['lemma']
    treenode.children = []
    return treenode

# This function takes a TokenTree as input,
# and returns a TokenList with the tokens from this tree
# and all its children/descendants,
# sorted by their ids.
# NOTE: It uses the same actual tokens, not copies!!
def tree_to_tokenlist(treenode):
    if len(treenode.children) == 0:
        return [treenode.token]
    else:
        answer = [treenode.token]
        for child in treenode.children:
            answer = answer + tree_to_tokenlist(child)
        return conllu.TokenList(sorted(answer, key = lambda x:x['id']))

# This function takes a TokenList as input,
# and returns a new TokenList,
# mostly the same as the input,
# but with new id's starting at 1 and increasing by 1 with each subsequent word,
# and all heads updated to match the new id's.
# Handy when you've been adding tokens and want to smooth everything out.
def reindex_tokenlist(sentence):
    sentence = copy.deepcopy(sentence)
    i = 0
    for token in sentence:
        i += 1
        if token['misc'] is None:
            token['misc'] = {}
        token['misc'].update(oldid = token['id'])
        token['id'] = i
    for token in sentence:
        newhead = 0
        for token2 in sentence:
            if token2['misc']['oldid'] == token['head']:
                newhead = token2['id']
        token['head'] = newhead
    for token in sentence:
        del token['misc']['oldid']
    return sentence

# MARK Functions to preprocess sentences, adding posited null determiners and such.

# This function is a hack,
# but it changes any "flat" dependency relations
# with head a NOUN and dependent a PROPN
# into "amod"...
# ... it just makes the denotations come out nicer.
# It takes a TokenList as input,
# and returns a new TokenList with the change.
def switch_propnflattoamod(sentence):
    sentence = copy.deepcopy(sentence)
    for token in sentence:
        if token['upos'] == 'PROPN' and token['deprel'] == 'flat':
            if sentence.filter(id=token['head'])[0]['upos'] == 'NOUN':
                token['deprel'] = 'amod'
    return sentence

# This function flattens a given relation.
# That is, it takes a TokenTree
# and a relation (a string corresponding to a value of the "deprel" field) as input.
# It returns a new TokenTree, just like the input, except
# anywhere there used to be that relation,
# - the dependent is completely flattened, with all _its_ dependents collapsed onto it in one token,
# - and then the dependent is concatenated to the head as part of that token, and removed.
# The result is that we essentially treat that relation as being between two parts of the same word,
# and combine the parts of that word,
# with POS and syntactic role corresponding to the relation's head.
def flatten_relation_tree(treenode,relation):
    treenode = copy.deepcopy(treenode)
    childrentokeep = []
    for child in treenode.children:
        if child.token['deprel'] == relation:
            child = concat_with_all_dependents(child)
            if child.token['id'] < treenode.token['id']:
                treenode.token['form'] = child.token['form'] + '_' + treenode.token['form']
                treenode.token['lemma'] = child.token['lemma'] + '_' + treenode.token['lemma']
            else:
                treenode.token['form'] = treenode.token['form'] + '_' + child.token['form']
                treenode.token['lemma'] = treenode.token['lemma'] + '_' + child.token['lemma']
        else:
            child = flatten_relation_tree(child, relation)
            childrentokeep.append(child)
    treenode.children = childrentokeep
    return treenode

# This function takes a TokenList as input
# and performs the same operation as flatten_relation_tree on it,
# returning the resulting TokenList with new indices.
def flatten_relation_list(sentence,relation):
    return reindex_tokenlist(
                tree_to_tokenlist(
                    flatten_relation_tree(
                        sentence.to_tree(), relation
                    )))


# This function takes a TokenList as input
# and returns a new TokenList,
# very similar to the first,
# but with null definite determiners inserted just before proper nouns without existing determiners,
# and null indefinite determiners inserted just before common nouns without existing determiners.
# Because of a hack involving the "flat" relation,
# no determiner is inserted if the noun is related to its head with the "amod" relation.
def add_nulldeterminers(sentence):
    sentence = copy.deepcopy(sentence)
    determinered = [sentence.filter(id=token['head'])[0] for token in sentence
                        if (token['upos']=='DET' and token['deprel']=='det')]
    determinerless = [token for token in sentence
                        if (token['upos'] in ('NOUN','PROPN')
                            and token not in determinered
                            and token['deprel'] != "amod")]
    for token in determinerless:
        sentence.append(conllu.models.Token({'id':token['id']-0.5,
                            'form':'∅-def' if token['upos']=='PROPN' else '∅-indf',
                            'lemma':'∅-def' if token['upos']=='PROPN' else '∅-indf',
                            'upos':'DET',
                            'xpos':None,
                            'feats':None,
                            'head':token['id'],
                            'deprel':'det',
                            'deps': None,
                            'misc': None
                        }))
    sentence = conllu.TokenList(sorted(sentence,key=lambda x:x['id']))
    return reindex_tokenlist(sentence)

# This combines all the previous steps in the correct order.
# So, it changes flat relations between Nouns and Proper Nouns to "amod" relations (for semantic reasons),
# flattens remaining "flat" relations,
# flattens remaining "compound" relations,
# and adds null determiners to any determiner-less nouns and proper nouns (except those modified by "amod"),
# returning a new TokenList
def preprocess(sentence):
    return add_nulldeterminers(
                flatten_relation_list(
                    flatten_relation_list(
                        switch_propnflattoamod(
                            sentence
                            ),
                        'flat'
                    ),
                    'compound'
                )
            )

# MARK all the word denotation templates, and relation denotations.

# Right now the program disregards punctuation entirely.
# The "simplifytyped" also disregards any relations it does not know, with a warning.
postemplates = {
    "ADJ":r'\x.([],[{}(x)])',
    "ADV":r'\x.([],[{}(x)])',
    "NOUN":r'\x.([],[{}(x)])',
    "PROPN":r'\x.([],[name(x,{})])',
    "VERB":r'\H.(([e],[{}(e)]) + H(e))',
    "PUNCT":r'([],[])',
    "ADP":r'\x.\y.([],[{}(y,x)])',
    "NUM":r'\x.([],[number(x,{})])',
}
detmeanings = {
    "a":DrtExpression.fromstring(r'\F.\G.(([x],[]) + F(x) + G(x))'),
    "the":DrtExpression.fromstring(r'\F.\G.(([x],[]) + F(x) + G(x))'),
    "some":DrtExpression.fromstring(r'\F.\G.(([x],[]) + F(x) + G(x))'),
    "every":DrtExpression.fromstring(r'\F.\G.([],[-(([x][-G(x)]) + F(x))])'),
    "all":DrtExpression.fromstring(r'\F.\G.([],[-(([x][-G(x)]) + F(x))])'),
    "each":DrtExpression.fromstring(r'\F.\G.([],[-(([x][-G(x)]) + F(x))])'),
    "∅-indf":DrtExpression.fromstring(r'\F.\G.(([x],[]) + F(x) + G(x))'),
    "∅-def":DrtExpression.fromstring(r'\F.\G.(([x],[]) + F(x) + G(x))')
}
relmeanings = {
    "nsubj":DrtExpression.fromstring(r'\F.\G.\H.F((\x.G((\y.(([],[nsubj(x,y)])+H(x))))))'),
    "obj":DrtExpression.fromstring(r'\F.\G.\H.F((\x.G((\y.(([],[obj(x,y)])+H(x))))))'),
    "amod":DrtExpression.fromstring(r'\F.\G.\x.(F(x)+G(x))'),
    "nummod":DrtExpression.fromstring(r'\F.\G.\x.(F(x)+G(x))'),
    "nmod":DrtExpression.fromstring(r'\F.\G.\x.(F(x)+G(x))'),
    "det":DrtExpression.fromstring(r'\F.\G.G(F)'),
    "advmod":DrtExpression.fromstring(r'\F.\G.\H.F((\x.(G(x) + H(x))))'),
    "obl":DrtExpression.fromstring(r'\F.\G.\H.F((\x.(G(x) + H(x))))'),
    "case":DrtExpression.fromstring(r'\F.\G.\x.F(\y.G(y,x))'),
    "ccomp":DrtExpression.fromstring(r'\F.\z.\H.F((\x.([p],[ccomp(x,p) p:z]) + H(x)))'),
    "csubj":DrtExpression.fromstring(r'\F.\z.\G.F((\x.([p],[csubj(x,p) p:z]) + G(x)))'),
    "root":DrtExpression.fromstring(r'\F.F((\x.([],[])))'),
}

# Some POS's and relations should be explicitly ignored.
POSwithnoden = ["CCONJ","PUNCT"]
relswithnoden = ["cc","conj","punct"]

# This function takes a Token as input
# and returns a new Token
# identical to the first, but with denotations added
# for both the word itself ("word_den") and its relation ("rel_den").
# It adds no denotation to punctuation Tokens,
# and adds a denotation based on one of the templates to any non-determiner word.
# For determiners and dependency relations,
# it simply matches to whatever denotation is stored for that word/relation.
def add_denotation(t):
    t = copy.deepcopy(t)
    if t['upos'] in POSwithnoden:
        return t
    elif t['upos'] == 'DET':
        if t['lemma'] in detmeanings.keys():
            t['word_den'] = detmeanings[t['lemma']]
        else:
            print("The word {} with ID {} is an unknown type of determiner.".format(t['form'],str(t['id'])))
    elif t['upos'] in postemplates.keys():
        t['word_den'] = DrtExpression.fromstring(postemplates[t['upos']].format(t['lemma']))
    else:
        print("The word {} with ID {} is a POS with unknown denotation.".format(t['form'],str(t['id'])))
    if t['deprel'] in relswithnoden:
        return t
    elif t['deprel'] in relmeanings.keys():
        t['rel_den'] = relmeanings[t['deprel']]
    else:
        print("The relation {} on the word with ID {} is a relation with unknown denotation.".format(t['deprel'],str(t['id'])))
    return t

# TODO add comment explaining this
# den1 and den2 are two denotations, and semtype is of course their semantic type.
def compute_conj(den1, den2, semtype):
    # There's a special proviso for semantic types ((et)t) and ((st)t)
    if semtype.like(SemType.fromstring('((ut)t)')):
        conjden = DrtExpression.fromstring(r'\F\G\H.(([x],[])+H(x)+F((\y.([],[partof(y,x)]))) + G((\y.([],[partof(y,x)]))))')
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
        return DrtExpression.fromstring(rf'\{xstr}.({recursivecallstr}+([{astr} {bstr}],[partof({astr},{xstr}) partof({bstr},{xstr})]))').simplify()
    else:
        x = DrtVariableExpression(unique_variable())
        xstr = str(x)
        recursivecallstr = str(compute_conj(den1(x).simplify(),den2(x).simplify(),semtype.get_right()))
        return DrtExpression.fromstring(rf'\{xstr}.{recursivecallstr}').simplify()

# MARK all the word and relation semantic types.

postypes = {
    "ADJ":SemType.fromstring('(et)'),
    "NUM":SemType.fromstring('(et)'),
    "ADV":SemType.fromstring('(st)'),
    "NOUN":SemType.fromstring('(et)'),
    "PROPN":SemType.fromstring('(et)'),
    "VERB":SemType.fromstring('((st)t)'),
    "PUNCT":SemType.fromstring('t'),
    "ADP":SemType.fromstring('(e(ut))'),
    "DET":SemType.fromstring('((et)((et)t))'),
}
reltypes = {
    "nsubj":CompositeType(SemType.fromstring('((st)t)'),
            CompositeType(SemType.fromstring('((et)t)'),
            SemType.fromstring('((st)t)'))),
    "obj":CompositeType(SemType.fromstring('((st)t)'),
            CompositeType(SemType.fromstring('((et)t)'),
            SemType.fromstring('((st)t)'))),
    "amod":CompositeType(SemType.fromstring('(et)'),SemType.fromstring('((et)(et))')),
    "nummod":CompositeType(SemType.fromstring('(et)'),SemType.fromstring('((et)(et))')),
    "nmod":CompositeType(SemType.fromstring('(et)'),
            CompositeType(SemType.fromstring('(et)'),
            SemType.fromstring('(et)'))),
    "det": CompositeType(SemType.fromstring('(et)'),
            CompositeType(SemType.fromstring('((et)((et)t))'),
            SemType.fromstring('((et)t)'))),
    "advmod":CompositeType(SemType.fromstring('((st)t)'),
            CompositeType(SemType.fromstring('(st)'),
            SemType.fromstring('((st)t)'))),
    "obl":CompositeType(SemType.fromstring('((st)t)'),
            CompositeType(SemType.fromstring('(st)'),
            SemType.fromstring('((st)t)'))),
    "case":CompositeType(SemType.fromstring('((et)t)'),
            CompositeType(SemType.fromstring('(e(ut))'),
            SemType.fromstring('(ut)'))),
    "ccomp":CompositeType(SemType.fromstring('((st)t)'),
            CompositeType(SemType.fromstring('?'),
            SemType.fromstring('((st)t)'))),
    "csubj":CompositeType(SemType.fromstring('((st)t)'),
            CompositeType(SemType.fromstring('?'),
            SemType.fromstring('((st)t)'))),
    "root":SemType.fromstring('(t(((st)t)t))'),
    # This is not the true type of conj, but what it has to be before simplify_node_typed runs.
    "conj":SemType.fromstring('(?(??))'), 
}

# Some POS's and relations should be explicitly ignored.
POSwithnotype = ["CCONJ","PUNCT"]
relswithnotype = ["cc","punct"]

# This function takes a Token as input
# and returns a new Token
# identical to the first, but with semantic types added
# for both the word itself ("word_type") and its relation ("rel_type").
# It adds no type to punctuation Tokens.
# Otherwise, it simply matches to whatever type is stored for that part-of-speech/relation.
def add_type(t):
    t = copy.deepcopy(t)
    if t['upos'] in POSwithnotype:
        return t
    elif t['upos'] in postypes.keys():
        t['word_type'] = postypes[t['upos']]
    else:
        print("The word {} with ID {} is a POS with unknown type.".format(t['form'],str(t['id'])))
    if t['deprel'] in relswithnotype:
        return t
    elif t['deprel'] in reltypes.keys():
        t['rel_type'] = reltypes[t['deprel']]
    else:
        print("The relation {} on the word with ID {} is a relation with unknown type.".format(t['deprel'],str(t['id'])))
    return t

# MARK actual semantic parsing

# This function takes three inputs:
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
# in the "word_den" field of the resulting TokenTree.
def simplifynodetyped(treenode):
    # Ignore nodes whose types we don't know.
    treenode = copy.deepcopy(treenode)
    if ('rel_type' not in treenode.token.keys()) or ('word_type' not in treenode.token.keys()):
        return treenode
    if len(treenode.children) > 0:
        # Depth-first: simplify children first.
        treenode.children = [simplifynodetyped(child) for child in treenode.children]
        # Compute binarizations
        starttype = treenode.token['word_type']
        endtype = treenode.token['rel_type'].get_right().get_left()
        iopairs = []
        usefulchildren = []
        for child in treenode.children:
            if 'rel_type' in child.token.keys() and 'word_type' in child.token.keys():
                iopairs.append((child.token['rel_type'].get_left(),child.token['rel_type'].get_right().get_right()))
                usefulchildren.append(child)
        binarizations = semtypebinarizations(starttype,endtype,iopairs,comp_func = lambda x,y:x.like(y))
        # Binarizations tell you which dependents to combine first.
        if binarizations:
            # TODO Change binarization code to allow for multiple binarizations.
            for i in binarizations[0]:
                child = usefulchildren[i]
                # TODO The next line is a hack; really should fix the DrtExpression code directly.
                child.token['word_den'] = child.token['word_den'].replace(DrtExpression.fromstring('x').variable,DrtExpression.fromstring('a'),True)
                # Treat conjunctions separately - it isn't computed as an ordinary lambda expression.
                if child.token['deprel'] == 'conj':
                    treenode.token['word_den'] = compute_conj(treenode.token['word_den'],child.token['word_den'],child.token['word_type'])
                else:
                    treenode.token['word_den'] = child.token['rel_den'](treenode.token['word_den'])(child.token['word_den']).simplify()
                treenode.token['word_type'] = child.token['rel_type'].get_right().get_right()
        elif iopairs:
            print("There was a problem in binarizing children of node {}".format(treenode.token['id']))
        treenode.children = []
    # If the node is one that's conjoined to another node,
    # we want the "conj" type to enforce that the other node has the same semantic type,
    # so we have to update its semantic type after simplifying this node.
    # It will change the "conj" from "(?(??))" to replace all the ?'s with the type of this node.
    if treenode.token['deprel'] == 'conj':
        treenode.token['rel_type'] = CompositeType(treenode.token['word_type'],
                                        CompositeType(treenode.token['word_type'],
                                        treenode.token['word_type']))
    # If the word has children and is an incompatible type,
    # the binarization will catch it.
    # but if it has no children, we still want the type to be right.
    if not treenode.token['word_type'].like(treenode.token['rel_type'].get_right().get_left()):
        print("The word type {} at the word with ID {} is incompatible with rel type {}".format(
            str(treenode.token['word_type']),str(treenode.token['id']),str(treenode.token['rel_type'])))
    # Root only takes one argument, and after we use it, we trash the relation.
    if treenode.token['deprel'] == 'root':
        treenode.token['word_den'] = treenode.token['rel_den'](treenode.token['word_den']).simplify()
    return treenode

# This function summarizes all the things we would want to do
# to an input file in the demos below.
# It takes the contents of a conllu, in string format, as input.
# It doesn't return anything, but prints the sentence expressed by the input file,
# and the computed denotation of the sentence.
# It's only here to streamline demos. 
def print_sentence_and_parse(testconllu):
    testsentence=conllu.parse(testconllu)[0]
    print(testsentence.metadata['text'])
    # Preprocess the sentence
    preprocessed = preprocess(testsentence)
    # Then add semantic information to each node...
    withdens = conllu.TokenList([add_denotation(token) for token in preprocessed])
    withtypes = conllu.TokenList([add_type(token) for token in withdens])
    # Then collapse all the nodes together!
    simplified = simplifynodetyped(withtypes.to_tree())
    simplified.token['word_den'].pretty_print()

# MARK Demos with a few Conll files

# basic transitive sentence
# We read the file, and print its structure...
with open("conllus\\the rider possesses the will.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# adjectives and binarization
with open("conllus\\the city enjoys a temperate climate.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# quantifiers
with open("conllus\\each section has a theme.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# obliques!
with open("conllus\\the first drops fell onto the parched stones from a cloudless blue sky.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# nominal modifiers
with open("conllus\\after a brief period in frankfurt the family moved to basel.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# determinerless nouns!
with open("conllus\\the first drops of rain fell onto the parched stones from a cloudless blue sky.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# determinerless nouns!
with open("conllus\\pages hung in tatters from the sodden blue spine.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# determinerless nouns!
with open("conllus\\successful stormtroopers share a high - five.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# determinerless nouns and numbers!
with open("conllus\\smith offered three reasons.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# proper nouns as det+noun, allowing them to be modified!
with open("conllus\\after a brief period in frankfurt the family moved to basel in switzerland.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# flat constructions!
with open("conllus\\wikinews interviews meteorological experts on cyclone phalin.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# flat and compound constructions, with special case for proper nouns
with open("conllus\\texas student ahmed mohamed inspires social movement.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# ccomp relation
# TODO note that the answer does not simplify because of the band-aid on the DRT module
with open("conllus\\the authors say the results confirm the existence of inadequate iodine intake in the australian population.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# csubj and ccomp
with open("conllus\\ending inflation means freeing all americans from the terror of runaway living costs.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# coordinate structures with nouns
with open("conllus\\most iodine in food comes from seafood milk and salt.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# coordinate structures with nouns and adjectives
with open("conllus\\fraud and corruption prevent a fair and proper expression of the public voice.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# coordinate structures with verb phrases
with open("conllus\\during the middle ages athens experienced a decline but re-emerged under byzantian rule.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# csubj and coordinate structures with adjectives (and null "mark" and "cop")
with open("conllus\\to continue this long trend is to guarantee tremendous social, cultural, political, and economic upheavals.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# coordinate structures with mismatched syntax
with open("conllus\\byron met and formed a close friendship with the younger john edleston.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)


# csubj and coordinate structures with mismatched syntax
with open("conllus\\practicing cultural relativism requires an open mind and a willingness to consider and even adapt to new values and norms.conll") as f:
    testconllu = f.read()
print_sentence_and_parse(testconllu)