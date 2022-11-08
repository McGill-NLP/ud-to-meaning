#from nltk.inference import TableauProver # helps resolve anaphora
from nltk.sem.drt import * # all the DRT things
from nltk.sem.logic import unique_variable # helps for manipulating DRT expressions
from pptree import Node # helps print traces nicely
from pptree import print_tree # helps print traces nicely
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
        if not token['misc']:
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
# into "nmod"...
# ... it just makes the denotations come out nicer.
# It takes a TokenList as input,
# and returns a new TokenList with the change.
def switch_propnflattonmod(sentence):
    sentence = copy.deepcopy(sentence)
    for token in sentence:
        if token['upos'] == 'PROPN' and token['deprel'] == 'flat':
            if sentence.filter(id=token['head'])[0]['upos'] == 'NOUN':
                token['deprel'] = 'nmod'
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
                        if (token['upos'] in ('NOUN','PROPN','PRON')
                            and token not in determinered
                            and token['deprel'] != "amod")]
    for token in determinerless:
        sentence.append(conllu.models.Token({'id':token['id']-0.5,
                            'form':'∅-def' if token['upos'] in ('PROPN','PRON') else '∅-indf',
                            'lemma':'∅-def' if token['upos'] in ('PROPN','PRON') else '∅-indf',
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
                        flatten_relation_list(
                            switch_propnflattonmod(
                                sentence
                            ),
                        'flat'
                        ),
                    'compound'
                    ),
                'goeswith'
                )
            )

# MARK all the word denotation templates, and relation denotations.
# Now they are all pairs of one SemType and one DRT expression.

# Right now the program disregards punctuation entirely.
# The "simplifytyped" also disregards any relations it does not know, with a warning.
postemplates = {
    "ADJ":[(SemType.fromstring('(e(st))'),r'\x.\y.([],[Attribute(x,y),{}(y)])')],
    "ADV":[(SemType.fromstring('(st)'),r'\x.([],[{}(x)])')],
    "NOUN":[(SemType.fromstring('(et)'),r'\x.([],[{}(x)])')],
    "PROPN":[(SemType.fromstring('(et)'),r'\x.([],[Name(x,{})])')],
    "PRON":[(SemType.fromstring('(et)'),r'\x.([],[])')],
    "VERB":[(SemType.fromstring('(j(st))'),r'\x.\y.([],[{}(y),Arg1(y,x)])'),
        (SemType.fromstring('(j(b(st)))'),r'\x.\y.\z.([],[{}(z),Arg1(z,x),Arg2(z,y)])'),
        (SemType.fromstring('(j(b(i(st))))'),r'\x.\y.\z.\w.([],[{}(w),Arg1(w,x),Arg2(w,y),Arg3(w,z)])'),
        (SemType.fromstring('(j(b(s(st))))'),r'\x.\y.\z.\w.([],[{}(w),Arg1(w,x),Arg2(w,y),Arg3(w,z)])'),
        (SemType.fromstring('(j(b(t(st))))'),r'\x.\y.\z.\w.([u],[{}(w),Arg1(w,x),Arg2(w,u),u:z])'),
        (SemType.fromstring('(j(s(st)))'),r'\x.\y.\z.([],[{}(z),Arg1(z,x),Arg2(z,y)])'),
        (SemType.fromstring('(j(t(st)))'),r'\x.\y.\z.([u],[{}(z),Arg1(z,x),Arg2(z,u),u:y])'),
        (SemType.fromstring('(j(i(t(st))))'),r'\x.\y.\z.\w.([u],[{}(w),Arg1(w,x),Arg2(w,u),Arg3(w,y),u:z])'),
    ],
    "ADP":[(SemType.fromstring('(e(et))'),r'\x.\y.([],[{}(y,x)])'),
            (SemType.fromstring('(e(st))'),r'\x.\y.([],[{}(y,x)])')],
    "NUM":[(SemType.fromstring('(et)'),r'\x.([],[Number(x,{})])'),
            (SemType.fromstring('((et)t)'),r'\F.(([x],[Name(x,{})]) + F(x))')
            ],
    "DET":[(SemType.fromstring('((et)((et)t))'),r'\F.\G.(([x],[]) + F(x) + G(x))'),
#            (SemType.fromstring('((et)((et)t))'),r'\F.\G.([],[-(([x][-G(x)]) + F(x))])'),
            ],
    "SCONJ":[(SemType.fromstring('(e(tt))'),r'\x.\y.([u],[{}(x,u),u:y])'),
    (SemType.fromstring('(s(st))'),r'\x.\y.([],[{}(x,y)])'),
    (SemType.fromstring('(t(tt))'),r'\x.\y.([u,v],[{}(u,v),u:x,v:y])'),
    ],
}

relmeanings = {
    "nsubj":[(CompositeType(SemType.fromstring('(j(st))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\y.G(\x.F(x,y))')),
            (CompositeType(SemType.fromstring('(j(b(st)))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(b(st))'))),
                DrtExpression.fromstring(r'\F.\G.\y.\z.G(\x.F(x,y,z))')),
            (CompositeType(SemType.fromstring('(j(i(st)))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(i(st))'))),
                DrtExpression.fromstring(r'\F.\G.\y.\z.G(\x.F(x,y,z))')),
            (CompositeType(SemType.fromstring('((et)t)'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('t'))),
                DrtExpression.fromstring(r'\F.\G.F(\x.(([z],[EQ(x,z)]) + G(\y.([],[EQ(y,z)]))))'))],
    "obl:agent":[(CompositeType(SemType.fromstring('(j(st))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\y.G(\x.F(x,y))')),
            (CompositeType(SemType.fromstring('(j(b(st)))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(b(st))'))),
                DrtExpression.fromstring(r'\F.\G.\y.\z.G(\x.F(x,y,z))')),
            (CompositeType(SemType.fromstring('(j(i(st)))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(i(st))'))),
                DrtExpression.fromstring(r'\F.\G.\y.\z.G(\x.F(x,y,z))'))],
    "obj":[(CompositeType(SemType.fromstring('(j(b(i(st))))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(j(i(st)))'))),
                DrtExpression.fromstring(r'\F.\G.\y.\z.\w.G(\x.F(y,x,z,w))')),
            (CompositeType(SemType.fromstring('(j(b(st)))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\y.\z.G(\x.F(y,x,z))')),
            (CompositeType(SemType.fromstring('(b(st))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\y.G(\x.F(x,y))'))],
    "nsubj:pass":[(CompositeType(SemType.fromstring('(j(b(i(st))))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(j(i(st)))'))),
                DrtExpression.fromstring(r'\F.\G.\y.\z.\w.G(\x.F(y,x,z,w))')),
            (CompositeType(SemType.fromstring('(j(b(st)))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\y.\z.G(\x.F(y,x,z))')),
            (CompositeType(SemType.fromstring('(j(st))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\y.G(\x.F(x,y))'))],
    "iobj":[(CompositeType(SemType.fromstring('(j(i(st)))'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\y.\z.G(\x.F(y,x,z))'))],
    "dislocated":[(CompositeType(SemType.fromstring('(st)'),
                CompositeType(SemType.fromstring('((et)t)'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\x.G(\y.(([],[Topic(x,y)])+F(x)))'))],
    "amod":[(CompositeType(SemType.fromstring('(et)'),
                        CompositeType(SemType.fromstring('(e(st))'),
                        SemType.fromstring('(et)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(([y],[]) + F(x) + G(x,y))'))],
    "nummod":[(CompositeType(SemType.fromstring('(et)'),
                CompositeType(SemType.fromstring('(et)'),
                SemType.fromstring('(et)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(F(x)+G(x))'))],
    "nmod":[(CompositeType(SemType.fromstring('(et)'),
                CompositeType(SemType.fromstring('(et)'),
                SemType.fromstring('(et)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(F(x)+G(x))'))],
    "det":[(CompositeType(SemType.fromstring('(et)'),
                CompositeType(SemType.fromstring('((et)((et)t))'),
                SemType.fromstring('((et)t)'))),
                DrtExpression.fromstring(r'\F.\G.G(F)'))],
    "advmod":[(CompositeType(SemType.fromstring('(st)'),
                CompositeType(SemType.fromstring('(st)'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(F(x)+G(x))')),
            (CompositeType(SemType.fromstring('(e(st))'),
                CompositeType(SemType.fromstring('(st)'),
                SemType.fromstring('(e(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(F(x,y)+G(y))'))],
    "obl":[(CompositeType(SemType.fromstring('(st)'),
                CompositeType(SemType.fromstring('(st)'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(F(x)+G(x))')),
            (CompositeType(SemType.fromstring('(e(st))'),
                CompositeType(SemType.fromstring('(st)'),
                SemType.fromstring('(e(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(F(x,y)+G(y))')),
            (CompositeType(SemType.fromstring('(et)'),
                CompositeType(SemType.fromstring('(et)'),
                SemType.fromstring('(et)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(F(x)+G(x))')),
            (CompositeType(SemType.fromstring('(st)'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(e(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(F(x)+G(x,y))'))],
    "case":[(CompositeType(SemType.fromstring('((et)t)'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\x.F(\y.G(y,x))')),
            (CompositeType(SemType.fromstring('((et)t)'),
                CompositeType(SemType.fromstring('(e(et))'),
                SemType.fromstring('(et)'))),
                DrtExpression.fromstring(r'\F.\G.\x.F(\y.G(y,x))')),
            (CompositeType(SemType.fromstring('((et)t)'),
                CompositeType(SemType.fromstring('(e(et))'),
                SemType.fromstring('((et)t)'))),
                DrtExpression.fromstring(r'\F.\G.F'))],
    "acl":[(CompositeType(SemType.fromstring('(et)'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(et)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(([y],[])+F(x)+G(x,y))')),
            (CompositeType(SemType.fromstring('(et)'),
                CompositeType(SemType.fromstring('(et)'),
                SemType.fromstring('(et)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(F(x)+G(x))'))],
    "acl:relcl":[(CompositeType(SemType.fromstring('(et)'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(et)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(([y],[])+F(x)+G(x,y))'))],
    "xcomp":[(CompositeType(SemType.fromstring('(j(b(s(st))))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(j(b(st)))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.\z.(([w],[])+F(x,y,w,z)+G(y,w))')),
            (CompositeType(SemType.fromstring('(j(b(t(st))))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(j(b(st)))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.\z.F(x,y,(([w],[]) + G(y,w)),z)')),
            (CompositeType(SemType.fromstring('(j(s(st)))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(([z],[]) + F(x,z,y) + G(x,z))')),
            (CompositeType(SemType.fromstring('(j(t(st)))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.F(x,(([w],[]) + G(x,w)),y)')),
                ],
    "ccomp":[(CompositeType(SemType.fromstring('(j(b(st)))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(([z,w],[])+F(x,z,y)+G(z,w))')),
            (CompositeType(SemType.fromstring('(j(t(st)))'),
                CompositeType(SemType.fromstring('(st)'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.F(x,(([z],[])+G(z)),y)')),
            (CompositeType(SemType.fromstring('(j(t(st)))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.F(x,(([z,w],[]) + G(z,w)),y)')),
            (CompositeType(SemType.fromstring('(j(i(t(st))))'),
                CompositeType(SemType.fromstring('(st)'),
                SemType.fromstring('(j(i(st)))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.\z.F(x,y,(([w],[]) + G(w)),z)'))],
    "csubj:pass":[(CompositeType(SemType.fromstring('(j(b(st)))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(([z,w],[])+F(x,z,y)+G(z,w))')),
            (CompositeType(SemType.fromstring('(j(t(st)))'),
                CompositeType(SemType.fromstring('(st)'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.F(x,(([z],[])+G(z)),y)')),
            (CompositeType(SemType.fromstring('(j(i(t(st))))'),
                CompositeType(SemType.fromstring('(st)'),
                SemType.fromstring('(j(i(st)))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.\z.F(x,y,(([w],[]) + G(w)),z)'))],
    "csubj":[(CompositeType(SemType.fromstring('(j(st))'),
                CompositeType(SemType.fromstring('(et)'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(([y],[]) + F(y,x) + G(y))')),
            (CompositeType(SemType.fromstring('(j(st))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(([y,w],[]) + F(y,x)+G(y,w))')),
            (CompositeType(SemType.fromstring('(j(st))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(([u],[u:(([y,w],[]) + G(y,w))]) + F(u,x))')),
            (CompositeType(SemType.fromstring('(s(st))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\A.\B.\x.(([y,z],[]) + F(z,x) + G(y,z))')),
            (CompositeType(SemType.fromstring('(s(st))'),
                CompositeType(SemType.fromstring('(st)'),
                SemType.fromstring('(st)'))),
                DrtExpression.fromstring(r'\F.\G.\x.(([y],[]) + F(y,x) + G(y))'))],
    "advcl":[(CompositeType(SemType.fromstring('(j(st))'),
                CompositeType(SemType.fromstring('(e(s(st)))'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(([z],[]) + F(x,y) + G(x,y,z))')),
            (CompositeType(SemType.fromstring('(j(st))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(([z],[]) + F(x,y) + G(x,z))')),
            (CompositeType(SemType.fromstring('(j(st))'),
                CompositeType(SemType.fromstring('(s(st))'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(([z],[]) + F(x,y) + G(y,z))')),
            (CompositeType(SemType.fromstring('(j(st))'),
                CompositeType(SemType.fromstring('(tt)'),
                SemType.fromstring('(j(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.G(F(x,y))')),
            (CompositeType(SemType.fromstring('(b(st))'),
                CompositeType(SemType.fromstring('(e(s(st)))'),
                SemType.fromstring('(b(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(([z],[]) + F(x,y) + G(x,y,z))')),
            (CompositeType(SemType.fromstring('(b(st))'),
                CompositeType(SemType.fromstring('(e(st))'),
                SemType.fromstring('(b(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(([z],[]) + F(x,y) + G(x,z))')),
            (CompositeType(SemType.fromstring('(b(st))'),
                CompositeType(SemType.fromstring('(s(st))'),
                SemType.fromstring('(b(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(([z],[]) + F(x,y) + G(y,z))')),
            (CompositeType(SemType.fromstring('(b(st))'),
                CompositeType(SemType.fromstring('(tt)'),
                SemType.fromstring('(b(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.G(F(x,y))')),
                ],
    "root":[(SemType.fromstring('(t((st)t))'),
                DrtExpression.fromstring(r'\F.(([x],[]) + F(x))')),
            (SemType.fromstring('(t(tt))'),
                DrtExpression.fromstring(r'\x.x'))],
    "mark":[(CompositeType(SemType.fromstring('(e(st))'),
                CompositeType(SemType.fromstring('(e(tt))'),
                SemType.fromstring('(et)'))),
                DrtExpression.fromstring(r'\F.\G.\x.G(x,(([y,z],[]) + F(y,z)))')),
            (CompositeType(SemType.fromstring('(st)'),
                CompositeType(SemType.fromstring('(e(tt))'),
                SemType.fromstring('(et)'))),
                DrtExpression.fromstring(r'\F.\G.\x.G(x,(([y],[]) + F(y)))')),
            (CompositeType(SemType.fromstring('(e(st))'),
                CompositeType(SemType.fromstring('(s(st))'),
                SemType.fromstring('(e(s(st)))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.\z.(F(x,z) + G(y,z))')),
            (CompositeType(SemType.fromstring('(st)'),
                CompositeType(SemType.fromstring('(s(st))'),
                SemType.fromstring('(s(st))'))),
                DrtExpression.fromstring(r'\F.\G.\x.\y.(F(y)+G(x,y))')),
            (CompositeType(SemType.fromstring('(st)'),
                CompositeType(SemType.fromstring('(t(tt))'),
                SemType.fromstring('(tt)'))),
                DrtExpression.fromstring(r'\F.\G.\x.G(x,(([y],[])+F(y)))')),
            (CompositeType(SemType.fromstring('t'),
                CompositeType(SemType.fromstring('(t(tt))'),
                SemType.fromstring('(tt)'))),
                DrtExpression.fromstring(r'\y.\G.\x.G(x,y)'))],
    # This is not the true type of conj, but what it has to be before simplify_node_typed runs.
    "conj":[(SemType.fromstring('(?(??))'),None)], 
}

# Some POS's and relations should be explicitly ignored.
POSwithnoden = ["CCONJ","PUNCT","INTJ","SYM","PART","X","AUX"]
relswithnoden = ["cc","punct","discourse",
                    "vocative","clf","dep",
                    "parataxis","reparandum",
                    "expl","expl:pass","aux:pass",
                    "list","expl:pv","aux",
                    "appos","cop",
                    "orphan","fixed"]

# This function takes a Token as input
# and returns a new Token
# identical to the first, but with denotations added
# for both the word itself ("word_den") and its relation ("rel_den").
# It adds no denotation to punctuation Tokens,
# and adds a denotation based on one of the templates to any non-determiner word.
def add_denotation(t):
    t = copy.deepcopy(t)
    if t['upos'] in POSwithnoden:
        return t
    elif t['upos'] in postemplates.keys():
        t['word_dens'] = [(template[0],DrtExpression.fromstring(template[1].format(t['lemma'].lower().replace("-","_"))))
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
        print("The relation {} on the word with ID {} is a relation with unknown denotation.".format(t['deprel'],str(t['id'])))
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

# MARK actual semantic parsing

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
                # TODO The next line is a hack; really should fix the DrtExpression code directly.
                if withtrace:
                    usefulchilddens = [(den[0],den[1].replace(DrtExpression.fromstring('x').variable,DrtVariableExpression(unique_variable()),True),den[2])
                                        for den in usefulchilddens]
                else:
                    usefulchilddens = [(den[0],den[1].replace(DrtExpression.fromstring('x').variable,DrtVariableExpression(unique_variable()),True))
                                        for den in usefulchilddens]
                # Treat conjunctions separately - it isn't computed as an ordinary lambda expression.
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
        print("There was a problem in binarizing children of node {}".format(treenode.token['id']))
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

# Turn a trace
# (output as a denotation of a sentence, from simplifynodetyped with withtrace=True)
# into a tree structure that pptree can print.
def tracetotree(t, parent = None, withdrs = False):
    if len(t) == 2:
        name = f"{t[0]}:{t[1]}" if withdrs else str(t[0])
        return Node(name,parent) if parent else Node(name)
    elif len(t) == 3:
        if t[2]['children']:
            # has children
            topnodename = ("[" if not parent else "") + f"{t[2]['form']}:{t[0]}" + (f":{t[1]}" if withdrs else "") + "]"
            topnode = Node(topnodename,parent) if parent else Node(topnodename)
            orignodename = f"[{t[2]['form']}:{t[2]['original'][0]}" + (f":{t[2]['original'][1]}" if withdrs else "")
            orignode = Node(orignodename,topnode)
            for child in t[2]['children']:
                relnodename = (f"{child[1][2]['deprel']}:" if len(child[1]) > 2 else "") + f"{child[0][0]}" + (f":{child[0][1]}" if (withdrs and child[0][1]) else "")
                relnode = Node(relnodename,orignode)
                childnode = tracetotree(child[1],relnode)
            return topnode
        else:
            # no children
            name = ("[" if not parent else "") + f"{t[2]['form']}:{t[0]}" + (f":{t[1]}" if withdrs else "") + "]"
            return Node(name,parent) if parent else Node(name)
    else:
        pass

# Actually printing the trace.
# TODO might be better to use print_tree2 rather than pptree, to preserve child order
def printtrace(t, withdrs = False, horizontal = True):
    print_tree(tracetotree(t,parent = None,withdrs = withdrs), horizontal = horizontal)